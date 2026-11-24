# Train a small luma CNN for live action

The best way to do real-time, in-browser upscaling of live-action streams is **not** to port an existing model. Build a short, MIT-clean shader pipeline around one small convolutional network that you train yourself. The pipeline is: clean-room deband at source resolution, then a **10k–50k-parameter luma-only CNN** that adds a residual on top of a cheap upscale and is trained on H.264/HEVC/AV1 degradations, then cheap chroma, then a light FidelityFX CAS/RCAS sharpen and synthetic grain at output resolution. All of it is hand-written WGSL, with a texture-packed WebGL2 fallback. Nothing ready-made fills the gap:

- **Every good real-time shader CNN was trained on anime.** That covers ArtCNN, CuNNy, Anime4K and FSRCNNX_LineArt.
- **The content-neutral classics are LGPL-3.0.** That covers RAVU, nnedi3, FSRCNNX and the SSim/Krig shaders.
- **The one model trained on live-action codec damage is non-commercial.** 2xLiveActionV1_SPAN is CC-BY-NC-SA.
- **Larger general models are orders of magnitude too slow.** Real-ESRGAN Compact and SPAN fall here.

The evidence that a tiny network is enough is strong. Amazon's 41k-parameter luma-only CNN gained about **+5.3 VMAF over Lanczos** on 500 kbps HEVC, and BasicVSR++ needs 7.3M parameters to beat it ([arXiv 2401.14641](https://arxiv.org/html/2401.14641v1)). WebSR shows that a network of about 10k parameters runs at **40–60 fps for 2x at 720p on mid-range integrated GPUs** in a browser ([free.upscaler.video](https://free.upscaler.video/technical/architecture/)). Until the custom weights exist, the performance tier can ship today with FSR1 EASU and RCAS (both MIT). The plan also has to work within hard constraints that no model choice fixes:

- **Tainted cross-origin video throws `SecurityError`.**
- **DRM video can never be read.**
- **Published browser timings are scarce.** The tier budgets below are anchored on WebSR's measurements and must be re-measured in your own benchmark.

## Anime-trained shaders and LGPL classics leave no ready-made pick

The mpv shader ecosystem, which the Anime4K plugin descends from, offers almost nothing that is both tuned for natural images and MIT. **ArtCNN** is MIT and has the best published scores: C4F32 reaches 43.47 dB PSNR on 2x luma, against 39.4–39.7 for RAVU-AR and 36.5 for Lanczos. But the README calls it "a collection of simple SISR models aimed at anime content". Its DS suffix means denoise+sharpen, not live action, and the only benchmark it has is a single anime frame that the author himself calls "very unscientific" ([ArtCNN](https://github.com/Artoriuz/ArtCNN); [Artoriuz](https://artoriuz.github.io/blog/mpv_upscaling.html)). **CuNNy** is "trained on visual novel screenshots/CG" and is LGPL-3.0 ([CuNNy](https://github.com/funnyplanter/CuNNy)).

The content-agnostic candidates are the ones r/mpv users actually recommend for real-life footage: **RAVU** (RAISR-style learned filters, where the `-ar` anti-ringing variants are "the recommended ones"), nnedi3 and FSRCNNX. All three are **LGPL-3.0**, and their weight tables are embedded in the shader source ([bjin/mpv-prescalers](https://github.com/bjin/mpv-prescalers); [FSRCNNX shader header](https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_8-0-4-1.glsl)). A WGSL port would be a derivative work and would stay LGPL. Magpie's HLSL ports are the most complete reference set in existence, but they are **GPL-3.0** and cannot be copied either ([Magpie](https://github.com/Blinue/Magpie)).

That leaves the two MIT vendor scalers. **AMD FSR 1** (EASU+RCAS, 2 passes, at most 0.4–1.0 ms at 4K output on RX 5700XT/RTX 2060S-class or faster) and **NVIDIA NIS** (1 pass, 6-tap directional, FP16 path, NV12 input) are both MIT ([GPUOpen](https://gpuopen.com/fidelityfx-superresolution/); [NIS](https://github.com/NVIDIAGameWorks/NVIDIAImageScaling)). Both vendors design these scalers for noise-free game frames. AMD says input "should be noise free" with film grain applied afterwards, and NVIDIA says to add grain after NIS "since sharpening amplifies grain". On compressed streams the sharpening stage is therefore the thing to restrain. Neither vendor scaler is a reason to skip a cleanup stage.

| Upscaler                         | Tuned for              | Cost class                   | License                    | Verdict for MIT live-action pack     |
| -------------------------------- | ---------------------- | ---------------------------- | -------------------------- | ------------------------------------ |
| FSR1 EASU+RCAS                   | Games, content-neutral | ≤0.4–1.0 ms @4K on dGPU      | MIT                        | **Ship** (performance tier)          |
| NIS                              | Games, content-neutral | 1 pass, no timings published | MIT                        | Optional alternative to FSR1         |
| Lanczos / EWA-jinc + AR          | Neutral                | Fixed taps                   | Textbook, write clean-room | **Ship** as CNN base/resampler       |
| ArtCNN C4F16 / C4F32             | Anime                  | ~12k / ~48k MAC/px           | MIT                        | Interim only; reuse the architecture |
| RAVU-AR, nnedi3, FSRCNNX         | Neutral / undocumented | LUT to ~10k MAC/px           | LGPL-3.0                   | **Avoid**                            |
| CuNNy, Ani4Kv2                   | Visual novels / anime  | Small CNN                    | LGPL-3.0 / CC BY-NC        | **Avoid**                            |
| Magpie ports of all of the above | —                      | —                            | GPL-3.0                    | **Never copy**                       |

## Neural SR models clear the license bar or the cost bar, rarely both

The models built for photographic content split cleanly into "permissive but too heavy" and "well-matched but restricted".

**Real-ESRGAN Compact** (realesr-general-x4v3, about 1.2M params) is BSD-3 for both code and weights, but it is natively 4x. By the notes' arithmetic it costs about **1.1 TMAC per 720p frame**. A WebGPU port measured **737 ms for a 200×133 image on an RTX 3070 Ti**, roughly 1000x slower than the Anime4K shaders ([Anime4K-WebGPU](https://github.com/Anime4KWebBoost/Anime4K-WebGPU); [Real-ESRGAN LICENSE](https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/LICENSE)).

**SPAN**, the NTIRE 2024 efficient-SR winner, is Apache-2.0 and has a native x2 checkpoint, but at about 0.4M params it ran at only **60 fps at 720p→1440p on a native RTX 2080**, against 429 fps for the 30k-parameter XLSR on the same benchmark ([fastforks/span](https://huggingface.co/fastforks/span); [arXiv 2602.11339](https://arxiv.org/html/2602.11339)). Browser overhead pushes that out of real time.

The NTIRE 2025 efficient-SR winners (0.13–0.17M params) are x4, bicubic-trained, and the report states no code license ([arXiv 2504.10686](https://arxiv.org/html/2504.10686v1)).

The OpenModelDB model that best matches streaming damage is **2xLiveActionV1_SPAN**. It was trained on JPEG, MPEG-4, H.264, VP9 and H.265 compression plus chroma subsampling, halos and scaling blur, and its author says it handles light-to-moderate grain. It is **CC-BY-NC-SA-4.0**, so it must not ship ([OpenModelDB](https://openmodeldb.info/models/2x-LiveActionV1-SPAN)). Helaman's **2x-NomosUni_span_multijpg(\_ldl)** models are CC-BY-4.0 and redistributable with attribution, but they are still SPAN-sized and JPEG-trained rather than video-trained ([OpenModelDB](https://openmodeldb.info/models/2x-NomosUni-span-multijpg-ldl)).

The useful finding is in the size class the browser can actually afford:

- **Qualcomm QuickSRNet-small** is 33k params and native 2x, tagged BSD-3. It makes "1080p outputs via 2x upscaling in 2.2 ms on a modern smartphone" ([arXiv 2303.04336](https://arxiv.org/abs/2303.04336v2); [HF](https://huggingface.co/qualcomm/QuickSRNetSmall)).
- **BVI-RTVSR** from Bristol/Netflix is **62k params** and ran at **0.8 ms on an RTX 3090** in the AV1-compressed AIM 2024 challenge, where it beat Lanczos on both PSNR and VMAF ([arXiv 2409.17256](https://arxiv.org/html/2409.17256v1)). Its weights and license are unknown.
- In the 2026 StreamSR benchmark, **fine-tuning tiny models on compressed YouTube content gave large gains**. A fine-tuned EfRLFN beat NVIDIA VSR at 37.85 vs 37.40 dB and 3.33 vs 2.57 subjective ([arXiv 2602.11339](https://arxiv.org/html/2602.11339)).

The lesson is consistent across the literature. At this size, **what the model was trained on matters more than its architecture**, and bicubic-only training, as in official SPAN, NTIRE and QuickSRNet, "does nothing" for blocking and tends to sharpen artifacts. You need your own weights trained on codec degradations, and that is also the only route to a clean license.

Two cautions shape the training target. In AIM 2024, SuperBicubic++ had the **highest VMAF (66.4) but the lowest PSNR (30.5)**, which shows that VMAF rewards sharpening ([arXiv 2409.17256](https://arxiv.org/html/2409.17256v1)). Real-ESRGAN-style general models also tend to smear grain into plastic texture, which is why `-dn` exists ([Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)). Train for fidelity with MAE loss, as the Amazon ARSR paper found MAE "removed the most artifacts", and validate against both PSNR and VMAF rather than chasing either one.

## Hand-written WGSL beats every ML runtime for per-frame work

Every project that achieves real-time browser SR hand-writes its convolutions. WebSR's author benchmarked his own WebGL shaders against TFLite. A 100k-parameter VDSR on a 360p image took **30 ms vs 1000 ms**, and the WebGPU version was a further 10–30% faster ([free.upscaler.video](https://free.upscaler.video/technical/architecture/)).

**ONNX Runtime Web's WebGPU EP** can keep tensors on the GPU through `Tensor.fromGpuBuffer` and `preferredOutputLocation: 'gpu-buffer'`, but no published SR video framerate exists for it ([ORT docs](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)). The only ORT-Web video SR project found runs offline ([Aliyun](https://developer.aliyun.com/article/1751533)). **WebNN is not an option**. Its origin trial started in Chrome 146 and was then **disabled again over "release blocking issues"** ([blink-dev](https://go.mail-archive.com/blink-dev@chromium.org/msg15933.html)). Use ORT-Web only as an accuracy oracle while you port a trained ONNX model to WGSL.

WebSR (MIT code) is the closest existing project and the best template for both backends:

- **WebGL2 path:** activations packed into RGBA textures, `mat4` uniforms mapping 4→4 channels, channel counts in multiples of 4.
- **WebGPU path:** compute shaders, f16 when `shader-f16` is present, tiling with padding, and an ImageBitmap fallback when `importExternalTexture` is missing ([websr](https://github.com/sb2702/websr)).

Its performance table is the only real browser measurement of live-action-class networks. For 2x at 720p:

| Hardware                | ~10k params | ~30k params | ~100k params |
| ----------------------- | ----------- | ----------- | ------------ |
| M4 MacBook Pro          | 120 fps     | 40 fps      | 15 fps       |
| Mid-range iGPU desktops | 40–60 fps   | —           | —            |
| $200 Chromebook         | ~40 fps     | —           | —            |

([free.upscaler.video](https://free.upscaler.video/technical/architecture/))

Do not lift WebSR's own `cnn-*`, `gan-*` and Anime4K `-rl` weights: they carry **no stated license**. Its training recipe is public and reusable, though: FFHQ, DIV2K, animation, UI and SD imagery, with blur, downscale and JPEG q70–90 with chroma subsampling.

Your existing plugin's frame plumbing already fits. It drives `copyExternalImageToTexture` from `requestVideoFrameCallback` and catches `SecurityError`, so keep that loop and optionally add a zero-copy `importExternalTexture` fast path, which is still "Limited availability" ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/importExternalTexture)). For hls.js streams the Gaggaoflix proxy must serve segments with `Access-Control-Allow-Origin`, and the video needs `crossOrigin="anonymous"`. Without that the plugin can only disable itself.

Browser coverage is good enough for a WebGPU-first design:

- **WebGPU on by default:** Chromium on Windows/macOS/ChromeOS/Android 12+, Firefox on Windows (141+) and macOS (145/147+), and Safari 26 on all Apple platforms.
- **Not shipped yet:** Firefox on Linux and Android ([gpuweb status, Oct 2026](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)).
- **shader-f16:** no per-browser table exists, so detect it at runtime.

Native RTX VSR and Edge VSR have no web API and only act on the `<video>` element itself ([Edge blog](https://blogs.windows.com/msedgedev/2023/03/08/video-super-resolution-in-microsoft-edge/)). That makes "off" a meaningful preset on RTX machines, not just a failure state.

## Cleanup decides perceived quality more than the upscaler does

For streaming sources, the order of operations is a hard rule. NVIDIA states that de-artifacting must happen before upscaling, or "compression artifacts can become even more visible when upscaling and sharpening" ([NVIDIA](https://blogs.nvidia.com/blog/ai-decoded-upscaling/)). RTX VSR itself predicts a **residual on top of a traditional upscale**, trained across compression levels against blockiness, ringing and banding ([NVIDIA](https://blogs.nvidia.com/blog/rtx-video-super-resolution)). That design is worth copying, because when the network is wrong the output degrades gracefully toward the base image.

The concrete stages:

- **Deband** is the cheapest high-value step. libplacebo's algorithm is 4 taps at a random angle per iteration, with the average replacing the pixel only below a threshold, then optional grain. Defaults are 1 iteration, threshold 3.0, radius 16, grain 4.0. The code is **LGPL-2.1+**, but the f3kdb-style behaviour is documented well enough for a clean-room MIT WGSL rewrite ([libplacebo options](https://libplacebo.org/options/); [sampling.c](https://github.com/haasn/libplacebo/blob/master/src/shaders/sampling.c)). Run it at source resolution, on luma at or below the default threshold, because live-action texture sits close to the threshold.
- **Classical deblocking** is not worth building. No mainstream real-time player ships one ([Magpie effects](https://github.com/Blinue/Magpie/wiki/Built-in%20effects)), and the trained CNN is where block and ringing removal belongs.
- **Chroma.** Luma-only processing is standard practice: ARSR runs its CNN on Y and bilinear on CbCr. ArtCNN's Chroma model scores best (43.68 dB), but KrigBilateral is LGPL-3.0+ and every chroma benchmark is on anime ([Artoriuz](https://artoriuz.github.io/blog/mpv_upscaling.html); [KrigBilateral](https://gist.github.com/igv/a015fc885d5c22e6891820ad89555637)). Use bilinear chroma by default and treat a chroma CNN as a quality-tier luxury.
- **Sharpening** should be FidelityFX **CAS** (MIT, one pass, contrast-adaptive) or RCAS at low strength, at output resolution ([GPUOpen CAS](https://gpuopen.com/fidelityfx-cas/)). Starting points are CAS ≈0.2–0.4 and RCAS ≈0.5–1.0 stops. These are the notes' inference and need tuning. Halos on skin are the first visible failure.
- **Grain** should be added last, at output resolution, as luma-only hashed noise. Grain added before upscaling turns blotchy. This mirrors AV1 Film Grain Synthesis, which denoises, codes and re-adds grain at the decoder ([Norkin & Birkbeck](https://norkin.org/pdf/DCC_2018_AV1_film_grain.pdf)), and it masks both deband smoothing and 8-bit banding.

## Three presets, one trained network family, and an auto ladder

The new package should mirror the existing plugin's `auto | off | performance | balanced | quality` contract, its median-of-samples benchmark, `pickAutoMode` and `stepDown`. It should add two things live action needs. The first is a **1x "cleanup only" path**, deband plus mild CAS plus grain, used when the source already matches the display. Today the plugin skips work below a 1.2x ratio. The second is a **frame-rate-aware budget**. The current fixed 8 ms default is conservative for 24 fps film, whose frame interval is 41.7 ms, but it leaves headroom for decode and compositing on iGPUs. A reasonable rule is min(12 ms, 35% of the frame interval). Treat that as a starting point to validate, not a measured optimum.

| Preset          | Pipeline (in order)                                                                                                                                                                                                                                             | Model                                                                         | Est. frame time, 720p→1440p iGPU (M4 / Iris-class)          | Est. frame time, 1080p→4K mid dGPU     | Backends                               |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------- | -------------------------------------- |
| **Performance** | RGB→Y'CbCr · deband (1 iter, luma) · FSR1 EASU to target · RCAS (soft) · grain                                                                                                                                                                                  | None (FSR1, MIT)                                                              | ~1–3 ms                                                     | ~1–2 ms (FSR1 ≤0.4–1.0 ms @4K per AMD) | WebGPU + WebGL2                        |
| **Balanced**    | deband · **luma CNN ~10–16k params, 2x residual over bilinear** · bilinear chroma · Lanczos resample to display · CAS 0.2–0.3 · grain                                                                                                                           | Custom "LA-S" (ArtCNN C4F16-class); interim: FSR1                             | ~8–25 ms (WebSR 10k: 120 fps M4, 40–60 fps mid iGPU)        | ~2–5 ms                                | WebGPU (f16 preferred) + WebGL2 packed |
| **Quality**     | deband · **luma CNN ~40–60k params, trained on codec damage** · optional chroma CNN or bilinear · Lanczos resample · CAS 0.2 · grain                                                                                                                            | Custom "LA-M" (ArtCNN C4F32 / ARSR-41k / QuickSRNet-medium / BVI-RTVSR class) | ~30+ ms, **not offered on iGPU** (WebSR 30k = 40 fps on M4) | ~5–10 ms                               | WebGPU only                            |
| **Auto**        | Benchmarks Quality → Balanced → Performance on the actual source/target size, picks the strongest under budget, steps down at runtime, caches per GPU+resolution; falls to 1x cleanup when no upscale is needed and to `off` when no CORS/DRM-free frame access | —                                                                             | —                                                           | —                                      | —                                      |

The frame times are extrapolations. The iGPU column scales WebSR's M4 and iGPU measurements, which work out to roughly 0.7–0.8 ms per thousand parameters for 2x at 720p on an M4. 1080p input costs 2.25x as much. The dGPU column uses FSR1's published timings, Anime4K-WebGPU's "within about 3 ms per 720p frame" on RTX 3070 Ti/4090, and the notes' MAC arithmetic, which puts a 12k-parameter network at about 50 GFLOP per 1080p frame ([Anime4K-WebGPU](https://github.com/Anime4KWebBoost/Anime4K-WebGPU)). **No WebGPU timing exists for any of these exact networks.** Build the benchmark harness, with timestamp queries on Iris Xe, an AMD APU, M-series and a 3060-class card, before you freeze the tier boundaries.

The implementation plan follows from the licensing and evidence picture:

- **Port now, as MIT WGSL:**
  - FSR1 EASU+RCAS and CAS from AMD's MIT sources. agyild's and atyuwen's GLSL gists are useful structural references.
  - Optionally NIS, using its coefficient tables from the MIT `NIS_Config.h`.
  - Clean-room Lanczos/bicubic, an f3kdb-style deband and hash grain.
  - Anime4K-WebGPU's MIT `Conv2d` and `DepthToSpace` WGSL helpers as the CNN substrate, which the sibling plugin already depends on through anime4k-webgpu-async.
  - For WebGL2, write the CNN in WebSR's RGBA-packed `mat4` style and constrain channel counts to multiples of 4 so one weight file serves both backends.
- **Train:**
  - Use one architecture family: ArtCNN's MIT C-architecture, or a plain re-parameterizable 3×3 stack in the QuickSRNet/ECBSR mould. Predict a 2x luma residual over bilinear, and train two widths, LA-S (~12k) and LA-M (~48k).
  - Degrade with LiveActionV1's published degradation recipe (reuse the recipe, not the weights). That means H.264, HEVC and AV1 at streaming bitrates/CRFs, 4:2:0, multi-pass scaling blur and halos, with mild grain _kept in the ground truth_ so the network learns not to erase it.
  - Use MAE loss, following ARSR's finding.
  - Validate on held-out live-action clips with PSNR-Y and VMAF, and run ORT-Web as the reference implementation to check the WGSL port's numerics.
  - Ship the weights under MIT, which you can do only because you trained them. The notes did not establish how each training dataset's license bears on that; check DIV2K/FFHQ terms before release.
  - Until LA-S exists, Balanced can fall back to FSR1 plus deband. ArtCNN C4F16 (MIT) can be an opt-in experimental model, labelled anime-trained.
- **Avoid:**
  - Any code or weights from bjin (RAVU, nnedi3), igv (FSRCNNX, SSim*, KrigBilateral), CuNNy, libplacebo or Magpie (LGPL/GPL).
  - 2xLiveActionV1_SPAN and Ani4Kv2 weights (NC).
  - WebSR's unlicensed weights, unless the author grants a license.
  - Real-ESRGAN Compact and SPAN as real-time tiers (too slow, though license-clean).
  - Building on WebNN.
  - If you ever bundle CC-BY Nomos weights as an "experimental" download, ship a NOTICE file. The package then becomes "MIT code + CC-BY weights", not pure MIT.

## Conclusion

The live-action problem differs from anime in kind, not only in degree. Anime upscalers win by reconstructing clean line art. Live-action streams are bottlenecked by codec damage, and the winning moves are a network that knows what H.264 and HEVC do to faces and foliage, and a pipeline that never sharpens before it cleans. That moves the project's centre of gravity from shader porting to a small, repeatable **training pipeline**. That is good news for licensing, because self-trained weights are the only fully MIT option, and the same training run is what improves quality the most.

The biggest open risk is empirical rather than architectural. Nobody has published a live-action shader comparison, an iGPU timing for any of these CNNs in WebGPU, or a viewer study of grain handling after real-time SR. The first deliverable of the new repo should therefore be the cross-GPU benchmark and an A/B harness on real Gaggaoflix streams, not the third preset. Two numbers from that harness decide the tier boundaries above: LA-S's millisecond cost on a typical Iris Xe, and whether LA-M visibly beats LA-S on 1080p sources.
