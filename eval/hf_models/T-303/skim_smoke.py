"""Pinned ESPnet SkiM example: whole-feature vs stateful frame inference + waveform decode."""
import sys
import time

from common import dummy, measure, run, torch_runtime

REPO = 'lichenda/wsj0_2mix_skim_small_causal'
REVISION = '40a67347c0d98183b924e3434a6f0dd68f3497fd'
ESP_REVISION = '3897ed8380bfb526d5c5dd2197eccbffbba7d8f8'
PREFIX = 'exp/enh_train_enh_skim_causal_small_raw/'


def smoke(hub):
    import yaml

    torch, device, sync = torch_runtime()
    config = yaml.safe_load(hub.get(REPO, PREFIX + 'config.yaml', REVISION).read_text())
    path = hub.get(REPO, PREFIX + '149epoch.pth', REVISION)
    # Fetch only the official inference modules, unmodified, into disposable scratch.
    files = ['layers/skim.py', 'layers/dprnn.py', 'layers/tcn.py', 'layers/adapt_layers.py',
             'encoder/conv_encoder.py', 'encoder/abs_encoder.py',
             'decoder/conv_decoder.py', 'decoder/abs_decoder.py']
    for name in files:
        hub.source('espnet2/enh/' + name, ESP_REVISION)
    sys.path.insert(0, str(hub.root / 'upstream'))
    from espnet2.enh.decoder.conv_decoder import ConvDecoder
    from espnet2.enh.encoder.conv_encoder import ConvEncoder
    from espnet2.enh.layers.skim import SkiM

    start = time.perf_counter()
    encoder = ConvEncoder(**config['encoder_conf']).to(device).eval()
    decoder = ConvDecoder(**config['decoder_conf']).to(device).eval()
    settings = config['separator_conf']
    skim = SkiM(input_size=encoder.output_dim, hidden_size=settings['unit'],
                output_size=encoder.output_dim * settings['num_spk'],
                num_blocks=settings['layer'], bidirectional=not settings['causal'],
                norm_type='cLN', segment_size=settings['segment_size'],
                seg_overlap=settings['seg_overlap'], mem_type=settings['mem_type']).to(device).eval()
    weights = torch.load(path, map_location=device, weights_only=True)
    for module, prefix in [(encoder, 'encoder.'), (decoder, 'decoder.'), (skim, 'separator.skim.')]:
        module.load_state_dict({k.removeprefix(prefix): v for k, v in weights.items()
                                if k.startswith(prefix)}, strict=True)
    sync()
    load_ms = (time.perf_counter() - start) * 1000
    rate = config['sample_rate']
    audio = torch.from_numpy(dummy(rate)[None]).to(device)
    lengths = torch.tensor([audio.shape[1]], device=device)

    def infer():
        features, _ = encoder(audio, lengths)
        logits = skim(features).reshape(1, -1, 2, encoder.output_dim)
        masked = features[:, :, None, :] * torch.relu(logits)
        return [decoder(masked[:, :, i, :], lengths)[0] for i in range(2)]

    with torch.inference_mode():
        result = measure(infer, synchronize=sync)
        features, _ = encoder(audio, lengths)
        # Follow SkiM's own __main__ example with 100 encoded frames (~50 ms),
        # spanning two segment boundaries and carrying the real trained state.
        features = features[:, :100]
        expected = skim(features)
        state = None
        frames = []
        sync()
        start = time.perf_counter()
        for index in range(features.shape[1]):
            output, state = skim.forward_stream(features[:, index:index + 1], state)
            frames.append(output)
        sync()
        stream_ms = (time.perf_counter() - start) * 1000
        actual = torch.cat(frames, dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        delta = (actual - expected).abs().max().item()
    return dict(result, load_ms=load_ms, device=device, sample_rate=rate, input_seconds=2,
                input_shapes=[list(audio.shape)], license='CC-BY-4.0',
                params=sum(p.numel() for m in [encoder, skim, decoder] for p in m.parameters()),
                stream_100_frames_ms=stream_ms, stream_max_abs_delta=delta,
                streaming_scope='separator features only; no live Stage/resampling adapter')


if __name__ == '__main__':
    run(REPO, smoke)
