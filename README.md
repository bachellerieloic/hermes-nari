# hermes-nari

Nari Labs speech for [Hermes Agent](https://github.com/NousResearch/hermes-agent): a plugin that adds
a `nari` text-to-speech provider and a `nari` speech-to-text provider, both backed by the hosted
[Nari Labs API](https://docs.narilabs.com/) and its free tier.

Once it is selected in `config.yaml`, every Hermes surface that speaks or listens uses it:

- Telegram voice notes are transcribed, and with `voice.auto_tts: true` replies come back as round
  voice bubbles;
- CLI voice mode and the wake word;
- Hermes-Relay on Android, hands-free;
- the dashboard, when its voice support lands.

No GPU, no local model download. Nari serves Qwen3 speech models; the `:free` variants cost nothing
within their daily allowance.

## Two-minute install

1. Create an API key at <https://app.narilabs.com/keys> and put it in the Hermes env file
   (`~/.hermes/.env`, or `$HERMES_HOME/.env`):

   ```
   NARI_API_KEY=your-key
   ```

2. Install the plugin. Pick one:

   ```bash
   # a) from Hermes, into ~/.hermes/plugins/nari (this repository ships a plugin.yaml at its root)
   hermes plugins install bachellerieloic/hermes-nari --enable

   # b) with pip, into the Python environment Hermes runs in (registers through the
   #    hermes_agent.plugins entry point)
   pip install git+https://github.com/bachellerieloic/hermes-nari

   # c) by hand: drop a checkout, or just the hermes_nari/ directory, into the plugins folder
   git clone https://github.com/bachellerieloic/hermes-nari ~/.hermes/plugins/nari
   ```

   Use one method only; two copies both named `nari` would fight over the registration.

3. Enable it and select it in `~/.hermes/config.yaml` (plugins are opt-in; `--enable` above already
   added the first block):

   ```yaml
   plugins:
     enabled:
       - nari

   tts:
     provider: nari
     providers:
       nari:
         model: qwen3-tts:free
         voice: leon
         output_format: ogg

   stt:
     provider: nari
     providers:
       nari:
         model: qwen3-asr:free
         language: en

   voice:
     auto_tts: true
   ```

4. Restart Hermes (the gateway too, if it runs as a service). `hermes-nari doctor` prints what the
   plugin resolved without touching the network; `hermes-nari selftest` speaks one sentence through
   Nari, transcribes it back and prints the timings.

Requirements: Python 3.11+, `ffmpeg` on PATH (the official Hermes Docker image ships it; it is
needed for mp3/ogg/opus/flac output and for every transcription, WAV output works without it), and
the `websockets` package (already in the Hermes environment).

## If you previously wired Nari as a command provider, delete that block

Hermes resolves a provider name in this order: built-in names, then `type: command` entries in
`config.yaml`, then plugins. A block like this one **shadows the plugin completely**, and Hermes keeps
running your old script instead:

```yaml
tts:
  providers:
    nari:
      type: command            # <- remove this whole entry
      command: python3 /path/to/nari-speak.py {input_path} {output_path} {voice}
```

The same applies to `stt.providers.nari` with `type: command`. Delete both, keep the plain settings
block shown above, restart. `hermes-nari doctor` warns when a shadowing block is still present.

## How settings resolve

For voice, model, output format and language, the most specific source wins:

1. the plugin's own block, `tts.providers.nari` / `stt.providers.nari` (Hermes's older layout
   `tts.nari` / `stt.nari` is accepted too);
2. what Hermes passes per call (it fills `voice`, `model`, `speed` and `output_format` from the
   top-level `tts` keys, and `stt.<provider>.language` / `stt.language` for transcription);
3. environment variables: `NARI_TTS_MODEL`, `NARI_TTS_VOICE`, `NARI_TTS_FORMAT`, `NARI_STT_MODEL`,
   `NARI_STT_LANGUAGE`, `NARI_BASE_URL`, `NARI_WS_URL`;
4. the defaults: `qwen3-tts:free`, `leon`, `mp3`, `qwen3-asr:free`, `en`.

Notes:

- `output_format` accepts what Hermes accepts: `mp3`, `wav`, `ogg`, `opus`, `flac`. Nari only returns
  WAV or raw PCM, so the plugin asks for PCM and converts with ffmpeg. `ogg` (Opus) is what Telegram
  voice bubbles want; the plugin always reports `voice_compatible`, so Hermes converts to Opus itself
  when you keep `mp3`.
- `language: auto` or an empty string means auto-detect for transcription. For speech the language
  belongs to the voice; a mismatching `language` is rejected by Nari, so the plugin never sends one.
- `speed` is ignored: Nari has no speech-rate control.
- Nari accepts at most 2,048 characters per request. Longer replies are split at sentence boundaries
  and the audio is joined, so nothing is truncated by the plugin. On Hermes 0.16 the dispatcher
  itself caps plugin providers at 4,000 characters; raise it with `tts.nari.max_text_length: 20000`
  if you need longer speeches. Newer Hermes chunks long text instead of truncating.

## Voices

```bash
hermes-nari voices                       # the configured model
hermes-nari voices --model qwen3-tts-fast:free
```

Voice ids are lowercase and case-sensitive (`leon`, `diana`, `claire`, ...). The current pack has
30 English voices and six Castilian Spanish ones (`gabriel`, `tomas`, `adrian`, `ines`, `alba`,
`elisa`); the voice decides the language. The catalog is per model, so list it for the model you use.

## Models, pricing, free tier

| Kind | Free | Partner (early access) |
| --- | --- | --- |
| Text to speech | `qwen3-tts:free`, `qwen3-tts-fast:free` | `qwen3-tts` USD 5 per million characters, `qwen3-tts-fast` USD 10 |
| Speech to text | `qwen3-asr:free`, `qwen3-asr-fast:free` | `qwen3-asr` USD 0.06 per audio hour, `qwen3-asr-fast` USD 0.12 |

Free models are best-effort: 2 concurrent requests or connections per organisation, and 100
accepted TTS requests and 100 accepted STT connections per model per day, resetting at 00:00 UTC.
A long reply that the plugin splits into several requests counts several times. Partner models
need an email to <founders@narilabs.com> during the public alpha. Details:
<https://docs.narilabs.com/models-and-pricing> and <https://docs.narilabs.com/rate-limits>.

## Compatibility

The provider interface was read from the Hermes source at v0.16.0 (tag `v2026.6.5`, June 2026)
and checked against the current plugin and TTS documentation and the 0.21 source:

- `TTSProvider`: `name`, `display_name`, `is_available`, `list_voices`, `list_models`,
  `default_model`, `default_voice`, `get_setup_schema`, `synthesize(text, output_path, *, voice,
  model, speed, format, **extra)`, `stream(...)`, `voice_compatible`.
- `TranscriptionProvider`: `name`, `display_name`, `is_available`, `list_models`, `default_model`,
  `get_setup_schema`, `transcribe(file_path, *, model, language, **extra)` returning the
  `{"success", "transcript", "provider", "error"}` envelope and never raising.
- Registration through `ctx.register_tts_provider` and `ctx.register_transcription_provider`;
  discovery through `~/.hermes/plugins/<name>/plugin.yaml` plus `__init__.py`, or the
  `hermes_agent.plugins` entry point.

If the base classes cannot be imported (a Hermes older than 0.16, or a future rename) the plugin
falls back to plain objects and logs a warning at registration instead of crashing Hermes.

`stream()` is implemented (PCM straight from Nari's streaming endpoint, transcoded on the fly) but,
as of Hermes 0.21, no Hermes surface calls a plugin's `stream()` yet; voice bubbles use `synthesize`.

## Docker

With the official image, `HERMES_HOME` is `/opt/data` inside the container and is usually mounted
from `~/.hermes` (or `/home/hermes/.hermes`) on the host. Install by hand from the host:

```bash
git clone https://github.com/bachellerieloic/hermes-nari /home/hermes/.hermes/plugins/nari
chown -R <hermes uid>:<hermes gid> /home/hermes/.hermes/plugins/nari
```

Then add the config blocks above to `/home/hermes/.hermes/config.yaml`, the key to
`/home/hermes/.hermes/.env`, and `docker compose restart` (or `docker restart hermes`). Inside the
container the same directory is `/opt/data/plugins/nari`.

## Command line

```bash
hermes-nari doctor      # key found? ffmpeg? resolved tts/stt settings, shadowing warnings
hermes-nari voices      # list voices for a model
hermes-nari selftest    # real API: speak, transcribe back, print timings (needs NARI_API_KEY)
python -m hermes_nari   # same thing without the console script
```

`selftest` uses one free TTS request and one free STT connection.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Replies are still spoken by the old script or by Edge | A `type: command` block named `nari` shadows the plugin (delete it), or the plugin is not in `plugins.enabled`, or Hermes was not restarted. `hermes plugins list` shows the state. |
| `NARI_API_KEY is not set` | Put the key in `~/.hermes/.env` (or `$HERMES_HOME/.env`) and restart. |
| `HTTP 401 ... INVALID_API_KEY` | Wrong or revoked key. |
| `HTTP 429 ... CONCURRENCY_LIMIT_EXCEEDED` | Two requests in flight already; wait and retry. |
| `HTTP 429 ... FREE_DAILY_LIMIT_EXCEEDED` | Allowance used up for that model until 00:00 UTC; switch to the `-fast` free model or ask for partner access. |
| `ffmpeg was not found on PATH` | Install ffmpeg, or point `HERMES_NARI_FFMPEG` at the binary. WAV output does not need it. |
| `INVALID_VOICE` | Voice ids are case-sensitive; `hermes-nari voices` lists them for the configured model. |
| Transcription returns `no transcript` | The note was silence or too short; Nari sends `transcript.completed` only for audio it understood. |
| Voice bubble arrives as a file on Telegram | Hermes converts to Opus with ffmpeg; check ffmpeg inside the Hermes environment. |

Logging: the plugin logs under `hermes_nari.*` through Hermes's logger (`hermes logs` or the gateway
log).

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

The tests run without network: an in-process fake of the Nari HTTP API and of the transcription
WebSocket, a fake `ffmpeg` on PATH, and stubs of the Hermes base classes when Hermes is not
installed (with Hermes on `PYTHONPATH` the real classes are used). Tests marked as needing the real
ffmpeg are skipped when it is absent.

## License

MIT. Not affiliated with Nari Labs or Nous Research.
