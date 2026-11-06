<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Golden quality at the v0.0.2-alpha snapshot cutoff

Current engine/help source is `1b0731a36b7482477b96d7a28b27c93e4dd49e3f`, release binary SHA256
`acdf429da22ae7a39504b897106e8cd56a22c3661bf086f60b5458ba5313f2af`.
The current GPU capture is **0/435 cases and 0/66,409 logical positions per
balanced format**. Current NLL, perplexity, KL, agreement and certain-agreement
metrics are unmeasured. No prior score is relabelled as a new measurement.

The current observer/capture producer passes 13 CPU parser/loop controls and all
117 embedded CUDA-payload comparisons. That qualifies CPU-side preparation,
not GPU replay, alignment or quality. The full 45-job plan (19 EXL3 and 26 PP5)
is deferred beyond this snapshot under the owner's 6 November deadline. It
covers the full 435-case target, including the 381 cases absent from the earlier
54-case subset. Additional teacher/cache/projection ablations are separate gates.

The complete prior dated metrics, methods, conditional-head denominator and
per-set tables remain in the historical measured tables below.
Those measurements cover 54/435 cases and 7,878/66,409 logical positions per
balanced format; exact is a sealed reference self-comparison, not a newly paired
current sample. Neither sampled SPARK success nor one-token Q restoration adds
a full-logit quality score. Full-context task quality remains unmeasured.

The serve-only TP4 prompt-chunk guard passes its separate 96/96 mixed identity
protocol and has **no independent full-logit golden score**. No baseline score
is relabelled as a guard score. Original balanced uses native PRO arithmetic,
so results depend on layer placement. Fast-axis tuples remain refused. Exact
runtime admission remains one session and 512 total tokens; long and concurrent
standard quality inputs are outside that admission.

See [the cutoff evidence](BENCHMARKS.md), the [remaining matrix
rows](COMPARISON_MATRIX.md), and [the scorer manual](../../tools/golden_score.md).

## Historical measured subset — dated evidence

# Golden metrics, 6 November 2026, tcb82

Release selection: **54 / 435 cases, 7878 / 66,409 logical positions, all 15 sets**. Select ceil(10% × set size), minimum one, using the lowest SHA256 of `matrix_6nov_v1/release_v1/` + case key. The fixed selection was saved before its new GPU jobs. Both balanced formats use exactly the same cases and all positions within each selected case. 381 unselected case pairs are the post-release queue; supplemental EXL3 captures from the intact first lease are retained outside the paired headline denominator.

Frozen scorer `golden_score`, SHA256005b91ccb651a6a4549aa7e977c0f52c36005390c265f34dc1edfa07f7cc8f4f; corpus epoch 4530cc637186edbe266947d7ce9960b0791c28b23be8d7efd067b001e8b0146c. Full 129,280-vocabulary little-endian F32 heads, F64 max-shifted log probabilities. NLL and KL(reference || candidate) are nats at probability temperature 1; PPL=exp(token-weighted mean NLL). Raw logit errors include common row offsets. Generation is greedy T0. Certain means golden top1–top2 margin strictly greater than 1 logit. Global RMS pools every logit/position. Logical case positions, including declared aliases, are weighted equally; this is greedy golden-continuation NLL, not an external held-out perplexity benchmark.

Balanced candidates use R’s qualified resident-model replay front end: each row independently prefills the complete golden prefix, then takes one native draw. Every observed token, position, sequence and rank is checked; complete heads must be finite and the native selected ID/F32 bits must equal the lowest-ID full-head argmax. The capture front end and observer are pinned against the frozen tcb82 engine. Balanced captures reserve context 16,384 (the admitted TP4 minimum); the exact source reserves 512. All scored prefixes are within 512. These are solo generate/THIN conditional metrics; multi-session BATCH distributions are not covered. This measures conditional-head quality, not a continuous-cache or autonomous free-running trajectory. First divergence below is the first zero-based conditional head disagreement; complete per-case indices are in the JSON receipts.

“Balanced” retains the qualified 5 Nov BF16/DSpark EXL3 or native-original launch base with tuning and pipeline defaults omitted. The literal four-axis all-balanced tuple refuses in tcb82 for both formats; its complete refusal text is archived. No fast/lossy axis tuple is admitted today. No held profile is substituted. †On original profiles the PRO runs native arithmetic; outputs depend on assigned layers. Exact is the admitted PP THIN serial M1 C1/512 reference.

**Prompt-chunk guard:** `--prefill_prompt_chunks=1` is serve-only and has **no independent full-logit golden score from this generate-based harness**. The measured balanced baseline above and the separately measured mixed-client identity are different evidence. No guard score or quality equivalence is inferred.

| Profile | Set | Cases | Positions | NLL nats | PPL | Top1 | Max &#124;Δlogit&#124; | RMS Δlogit | KL mean | KL p95 | KL max | Certain agreement | Uncertain agreement | Cases with first divergence | Earliest conditional index |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TP4 EXL3 balanced | OVERALL | 54 | 7878 | 0.11461817851639454 | 1.1214451639588234 | 7627/7878 (96.81391216044682%) | 60.429954528808594 | 0.7090969785404917 | 0.03291790405086929 | 0.14159738620395465 | 3.034369790019401 | 7202/7277 (98.96935550364161%) | 425/601 (70.71547420965058%) | 39 | 0 |
| TP4 EXL3 balanced | calibration | 1 | 1 | 4.5056952042389574e-08 | 1.000000045056953 | 1/1 (100.0%) | 5.027801990509033 | 1.028551447841605 | 7.623019135234047e-09 | 7.623019135234047e-09 | 7.623019135234047e-09 | 1/1 (100.0%) | n=0 | 0 | none |
| TP4 EXL3 balanced | curtis_teacher | 2 | 26 | 0.07888175094339883 | 1.0820763599960266 | 25/26 (96.15384615384616%) | 6.513946056365967 | 0.7206778973074532 | 0.034870544737599735 | 0.010570204094726709 | 0.8894058995330687 | 25/26 (96.15384615384616%) | n=0 | 1 | 2 |
| TP4 EXL3 balanced | donor_coherence | 1 | 27 | 0.23673138490531181 | 1.2671007096808438 | 24/27 (88.88888888888889%) | 7.501242160797119 | 0.9200072291027306 | 0.1756832152226359 | 0.7977089405113665 | 1.005194545532284 | 22/23 (95.65217391304348%) | 2/4 (50.0%) | 1 | 3 |
| TP4 EXL3 balanced | donor_decode | 1 | 400 | 0.20011455699358924 | 1.2215426864028327 | 384/400 (96.0%) | 7.548598766326904 | 0.49098443861408325 | 0.02484134965280895 | 0.11982971679262738 | 0.590472067287142 | 332/332 (100.0%) | 52/68 (76.47058823529412%) | 1 | 77 |
| TP4 EXL3 balanced | donor_health | 1 | 8 | 0.005608088871029739 | 1.0056238436390332 | 8/8 (100.0%) | 3.9401508569717407 | 0.5587185189750142 | 0.001047705532274593 | 0.00410697702511138 | 0.005463263242709772 | 8/8 (100.0%) | n=0 | 0 | none |
| TP4 EXL3 balanced | donor_warmup | 1 | 32 | 0.03271716847774563 | 1.0332582599038185 | 32/32 (100.0%) | 4.303142309188843 | 0.5672692835274208 | 0.006329154964012717 | 0.022404524292040193 | 0.12537670220045022 | 31/31 (100.0%) | 1/1 (100.0%) | 0 | none |
| TP4 EXL3 balanced | miaai | 1 | 400 | 0.020299390313819717 | 1.0205068241533182 | 398/400 (99.5%) | 20.661359786987305 | 0.724960097799931 | 0.004528231565379571 | 0.004139789668834447 | 0.4235541929026917 | 394/394 (100.0%) | 4/6 (66.66666666666667%) | 1 | 90 |
| TP4 EXL3 balanced | miaai_warmup | 1 | 32 | 3.546008778167002e-07 | 1.0000003546009406 | 32/32 (100.0%) | 5.167304992675781 | 0.6205913485598237 | 2.569256727659359e-07 | 2.069726264774902e-07 | 7.554412224777048e-06 | 32/32 (100.0%) | n=0 | 0 | none |
| TP4 EXL3 balanced | mixed16 | 1 | 165 | 0.024618654693680665 | 1.0249241959590802 | 164/165 (99.39393939393939%) | 11.295315265655518 | 0.8184323072098049 | 0.011332767644274842 | 0.0397491960330468 | 0.6168325439318044 | 163/163 (100.0%) | 1/2 (50.0%) | 1 | 0 |
| TP4 EXL3 balanced | mixed_calibration | 1 | 1 | 1.1817903757973496e-06 | 1.0000011817910741 | 1/1 (100.0%) | 4.916481465101242 | 0.9616860604496027 | 3.509591092545504e-07 | 3.509591092545504e-07 | 3.509591092545504e-07 | 1/1 (100.0%) | n=0 | 0 | none |
| TP4 EXL3 balanced | q_harness | 1 | 1 | 0.7483075455674258 | 2.11342012081058 | 1/1 (100.0%) | 7.534449577331543 | 1.4640995358456035 | 0.059695590857471656 | 0.059695590857471656 | 0.059695590857471656 | n=0 | 1/1 (100.0%) | 0 | none |
| TP4 EXL3 balanced | r_reference | 1 | 16 | 0.03651809569358126 | 1.0371930725759602 | 16/16 (100.0%) | 6.330798149108887 | 0.7900446230953895 | 0.02056857332125686 | 0.10210028847692319 | 0.1624112514605268 | 15/15 (100.0%) | 1/1 (100.0%) | 0 | none |
| TP4 EXL3 balanced | spark | 20 | 3351 | 0.11158424100267386 | 1.1180479255167857 | 3250/3351 (96.9859743360191%) | 60.429954528808594 | 0.7187239447076962 | 0.02881884672368427 | 0.12121138022813598 | 3.034369790019401 | 3076/3101 (99.19380844888745%) | 174/250 (69.6%) | 16 | 1 |
| TP4 EXL3 balanced | spark_greedy_proposals | 20 | 3354 | 0.12345100178996517 | 1.1313945668752965 | 3233/3354 (96.39236732259988%) | 50.12538528442383 | 0.7071870313468344 | 0.04009837268050754 | 0.19509025744325792 | 2.750185283961225 | 3048/3091 (98.60886444516338%) | 185/263 (70.34220532319392%) | 17 | 1 |
| TP4 EXL3 balanced | spark_warmup | 1 | 64 | 0.18584070517600035 | 1.2042304174258418 | 58/64 (90.625%) | 6.895256042480469 | 0.9686730165524193 | 0.13121585470597946 | 0.7175945422392948 | 1.5664007933266522 | 54/59 (91.52542372881356%) | 4/5 (80.0%) | 1 | 1 |
| Original balanced† | OVERALL | 54 | 7878 | 0.09127083981494669 | 1.095565687888087 | 7737/7878 (98.21020563594821%) | 59.28239631652832 | 0.44146871822991723 | 0.010350991411186521 | 0.039149578253079784 | 3.115546992486973 | 7263/7277 (99.80761302734643%) | 474/601 (78.86855241264558%) | 35 | 0 |
| Original balanced† | calibration | 1 | 1 | 2.8322539467869266e-08 | 1.0000000283225399 | 1/1 (100.0%) | 2.372323513031006 | 0.4903835076696238 | 2.0618152464506387e-09 | 2.0618152464506387e-09 | 2.0618152464506387e-09 | 1/1 (100.0%) | n=0 | 0 | none |
| Original balanced† | curtis_teacher | 2 | 26 | 0.01354057223013794 | 1.0136326609535395 | 26/26 (100.0%) | 3.72194766998291 | 0.46350524798392384 | 0.0006937264357656478 | 0.004248747289371866 | 0.004620420780014301 | 26/26 (100.0%) | n=0 | 0 | none |
| Original balanced† | donor_coherence | 1 | 27 | 0.12883951734209426 | 1.1375075592803732 | 27/27 (100.0%) | 4.072870254516602 | 0.46820749389511235 | 0.01948554336869497 | 0.0540931887375403 | 0.22869708033740077 | 23/23 (100.0%) | 4/4 (100.0%) | 0 | none |
| Original balanced† | donor_decode | 1 | 400 | 0.18527220382319548 | 1.2035460053674703 | 384/400 (96.0%) | 7.871542930603027 | 0.325392077399269 | 0.007531668222201987 | 0.035641728847888 | 0.13481030727366133 | 332/332 (100.0%) | 52/68 (76.47058823529412%) | 1 | 9 |
| Original balanced† | donor_health | 1 | 8 | 0.002702100270073071 | 1.0027057542333917 | 8/8 (100.0%) | 2.6975460052490234 | 0.34114144486132164 | 0.00029321531357500466 | 0.001311491299922661 | 0.0018899403345986991 | 8/8 (100.0%) | n=0 | 0 | none |
| Original balanced† | donor_warmup | 1 | 32 | 0.05707048586209247 | 1.0587304332928538 | 31/32 (96.875%) | 3.0395126342773438 | 0.3678435972839547 | 0.0035676886416310025 | 0.01825534225775256 | 0.06106429652946865 | 31/31 (100.0%) | 0/1 (0.0%) | 1 | 9 |
| Original balanced† | miaai | 1 | 400 | 0.016685349497530567 | 1.0168253273844112 | 398/400 (99.5%) | 13.7462797164917 | 0.5241550645275672 | 0.0007538136230608645 | 0.0006304247182421291 | 0.09862525877262722 | 394/394 (100.0%) | 4/6 (66.66666666666667%) | 1 | 90 |
| Original balanced† | miaai_warmup | 1 | 32 | 5.5270810046039885e-08 | 1.0000000552708115 | 32/32 (100.0%) | 3.9780007004737854 | 0.4439198045703311 | 1.0380617284731481e-08 | 3.061780369780046e-08 | 2.1771432448750928e-07 | 32/32 (100.0%) | n=0 | 0 | none |
| Original balanced† | mixed16 | 1 | 165 | 0.015830383241872904 | 1.015956347568742 | 163/165 (98.78787878787878%) | 5.68071985244751 | 0.4702217572514428 | 0.0018431966155138628 | 0.008642806713950409 | 0.05811032081134434 | 163/163 (100.0%) | 0/2 (0.0%) | 1 | 0 |
| Original balanced† | mixed_calibration | 1 | 1 | 1.0284476040257497e-06 | 1.0000010284481329 | 1/1 (100.0%) | 2.099597930908203 | 0.46088601819456654 | 8.755819314264073e-08 | 8.755819314264073e-08 | 8.755819314264073e-08 | 1/1 (100.0%) | n=0 | 0 | none |
| Original balanced† | q_harness | 1 | 1 | 2.498596984719599 | 12.165413720244192 | 0/1 (0.0%) | 4.058980464935303 | 0.8696941546683566 | 0.8586091947992732 | 0.8586091947992732 | 0.8586091947992732 | n=0 | 0/1 (0.0%) | 1 | 0 |
| Original balanced† | r_reference | 1 | 16 | 0.036522600758316884 | 1.0371977452084207 | 16/16 (100.0%) | 3.2435388565063477 | 0.47433367049945063 | 0.009122921495028314 | 0.051577695843995866 | 0.1007669862487477 | 15/15 (100.0%) | 1/1 (100.0%) | 0 | none |
| Original balanced† | spark | 20 | 3351 | 0.08968475229404009 | 1.0938294021367698 | 3300/3351 (98.47806624888094%) | 25.49782371520996 | 0.43377383497014754 | 0.00906675054243261 | 0.03074646721496145 | 3.115546992486973 | 3098/3101 (99.9032570138665%) | 202/250 (80.8%) | 14 | 8 |
| Original balanced† | spark_greedy_proposals | 20 | 3354 | 0.09364682560573394 | 1.0981718312491404 | 3290/3354 (98.09183064997019%) | 59.28239631652832 | 0.44750728361682945 | 0.012791062334433745 | 0.05724135095805972 | 1.4819703621630378 | 3082/3091 (99.70883209317373%) | 208/263 (79.08745247148289%) | 15 | 5 |
| Original balanced† | spark_warmup | 1 | 64 | 0.1914066853395968 | 1.2109518283089478 | 60/64 (93.75%) | 5.738931655883789 | 0.5087389636323338 | 0.04652580222554803 | 0.24979259423874042 | 1.1267557122054725 | 57/59 (96.61016949152543%) | 3/5 (60.0%) | 1 | 16 |
| Original exact† | OVERALL | 54 | 7878 | 0.0800008375773364 | 1.0832879750120352 | 7878/7878 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7277/7277 (100.0%) | 601/601 (100.0%) | 0 | none |
| Original exact† | calibration | 1 | 1 | 2.889500619216982e-08 | 1.0000000288950066 | 1/1 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/1 (100.0%) | n=0 | 0 | none |
| Original exact† | curtis_teacher | 2 | 26 | 0.017145480708768066 | 1.0172933081119304 | 26/26 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 26/26 (100.0%) | n=0 | 0 | none |
| Original exact† | donor_coherence | 1 | 27 | 0.12985019784518265 | 1.1386577971562395 | 27/27 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 23/23 (100.0%) | 4/4 (100.0%) | 0 | none |
| Original exact† | donor_decode | 1 | 400 | 0.17912383086864106 | 1.1961688576258664 | 400/400 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 332/332 (100.0%) | 68/68 (100.0%) | 0 | none |
| Original exact† | donor_health | 1 | 8 | 0.0027902745388500893 | 1.0027941709780535 | 8/8 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8/8 (100.0%) | n=0 | 0 | none |
| Original exact† | donor_warmup | 1 | 32 | 0.04755269599127863 | 1.0487014620283552 | 32/32 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 31/31 (100.0%) | 1/1 (100.0%) | 0 | none |
| Original exact† | miaai | 1 | 400 | 0.015154879941824214 | 1.015270297443903 | 400/400 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 394/394 (100.0%) | 6/6 (100.0%) | 0 | none |
| Original exact† | miaai_warmup | 1 | 32 | 3.633166298867831e-08 | 1.0000000363316637 | 32/32 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32/32 (100.0%) | n=0 | 0 | none |
| Original exact† | mixed16 | 1 | 165 | 0.014617770778986195 | 1.0147251328839348 | 165/165 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 163/163 (100.0%) | 2/2 (100.0%) | 0 | none |
| Original exact† | mixed_calibration | 1 | 1 | 7.994888603787769e-07 | 1.00000079948918 | 1/1 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1/1 (100.0%) | n=0 | 0 | none |
| Original exact† | q_harness | 1 | 1 | 0.5592030423872207 | 1.7492778443341668 | 1/1 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | n=0 | 1/1 (100.0%) | 0 | none |
| Original exact† | r_reference | 1 | 16 | 0.044640147167022054 | 1.0456515115390297 | 16/16 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 15/15 (100.0%) | 1/1 (100.0%) | 0 | none |
| Original exact† | spark | 20 | 3351 | 0.07797972255409016 | 1.0811007364861138 | 3351/3351 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3101/3101 (100.0%) | 250/250 (100.0%) | 0 | none |
| Original exact† | spark_greedy_proposals | 20 | 3354 | 0.08217862798891533 | 1.085649719936286 | 3354/3354 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3091/3091 (100.0%) | 263/263 (100.0%) | 0 | none |
| Original exact† | spark_warmup | 1 | 64 | 0.10026280752535738 | 1.1054614034789272 | 64/64 (100.0%) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 59/59 (100.0%) | 5/5 (100.0%) | 0 | none |

The exact reference’s full 435-case / 66,409-position self-score: NLL 0.07415614837229319, PPL 1.0769749602543766, top1 66409/66409, Δmax/RMS 0/0 and KL mean/p95/max 0/0/0. This is reference self-comparison. Its instrumented capture wall time is not a speed result; exact speed comes only from stage3 uninstrumented real-server runs.

| Set | Available cases | Selected cases | Selected positions |
| --- | --- | --- | --- |
| calibration | 1 | 1 | 1 |
| curtis_teacher | 12 | 2 | 26 |
| donor_coherence | 3 | 1 | 27 |
| donor_decode | 3 | 1 | 400 |
| donor_health | 1 | 1 | 8 |
| donor_warmup | 3 | 1 | 32 |
| miaai | 3 | 1 | 400 |
| miaai_warmup | 3 | 1 | 32 |
| mixed16 | 2 | 1 | 165 |
| mixed_calibration | 1 | 1 | 1 |
| q_harness | 1 | 1 | 1 |
| r_reference | 3 | 1 | 16 |
| spark | 198 | 20 | 3351 |
| spark_greedy_proposals | 198 | 20 | 3354 |
| spark_warmup | 3 | 1 | 64 |

First conditional divergence, zero-based generated position (`none` means no disagreement on the scored prefixes):

| Case | EXL3 balanced | Original balanced | Exact reference |
| --- | --- | --- | --- |
| calibration/reply_ok | none | none | none |
| curtis_teacher/latest_owner_after_transfer-short-wrong | none | none | none |
| curtis_teacher/read_only_scope-short-gold | 2 | none | none |
| donor_coherence/sky_blue | 3 | none | none |
| donor_decode/prose | 77 | 9 | none |
| donor_health/prose_after | none | none | none |
| donor_warmup/prose | none | 9 | none |
| miaai/code | 90 | 90 | none |
| miaai_warmup/structured | none | none | none |
| mixed16/translate_notice | 0 | 0 | none |
| mixed_calibration/tagged_reply_ok | none | none | none |
| q_harness/reply_ready | none | 0 | none |
| r_reference/review_chat | none | none | none |
| spark/p2_math_c2_s1 | 12 | 93 | none |
| spark/p2_prose_c8_s4 | 19 | 25 | none |
| spark/p2_reasoning_c8_s4 | 134 | 10 | none |
| spark/p2_math_c8_s1 | 12 | 194 | none |
| spark/p1_narrative_c8_s1 | 1 | 8 | none |
| spark/p2_ceiling_count_c8_s6 | none | none | none |
| spark/p2_json_c8_s3 | 7 | 47 | none |
| spark/p1_reasoning_c8_s5 | 10 | 115 | none |
| spark/p1_coding_c8_s0 | 113 | none | none |
| spark/p2_json_c8_s2 | 7 | 12 | none |
| spark/p1_coding_c8_s7 | 113 | none | none |
| spark/p2_coding_c8_s2 | 113 | 192 | none |
| spark/p2_narrative_c8_s6 | 1 | 12 | none |
| spark/p1_math_c2_s1 | 12 | 118 | none |
| spark/p1_reasoning_c2_s0 | 115 | 10 | none |
| spark/p1_json_c8_s0 | 7 | 12 | none |
| spark/p2_reasoning_c8_s5 | 5 | 72 | none |
| spark/p2_format_c8_s7 | none | none | none |
| spark/p2_ceiling_count_c2_s0 | none | none | none |
| spark/p1_format_c1_s0 | none | none | none |
| spark_greedy_proposals/p1g_coding_c2_s1 | 113 | none | none |
| spark_greedy_proposals/p2g_json_c8_s7 | 7 | 59 | none |
| spark_greedy_proposals/p2g_ceiling_count_c8_s2 | none | none | none |
| spark_greedy_proposals/p2g_coding_c1_s0 | 113 | 194 | none |
| spark_greedy_proposals/p1g_reasoning_c8_s1 | 95 | 5 | none |
| spark_greedy_proposals/p2g_math_c8_s4 | 12 | 118 | none |
| spark_greedy_proposals/p1g_reasoning_c8_s0 | 162 | 198 | none |
| spark_greedy_proposals/p1g_math_c2_s0 | 12 | 5 | none |
| spark_greedy_proposals/p1g_reasoning_c8_s2 | 115 | 115 | none |
| spark_greedy_proposals/p2g_narrative_c8_s4 | 1 | 16 | none |
| spark_greedy_proposals/p1g_ceiling_count_c8_s7 | none | none | none |
| spark_greedy_proposals/p1g_json_c8_s5 | 7 | 12 | none |
| spark_greedy_proposals/p2g_summary_c8_s6 | 10 | 10 | none |
| spark_greedy_proposals/p2g_narrative_c8_s0 | 1 | 8 | none |
| spark_greedy_proposals/p2g_prose_c2_s0 | 19 | 25 | none |
| spark_greedy_proposals/p1g_prose_c8_s3 | 25 | 25 | none |
| spark_greedy_proposals/p1g_prose_c8_s2 | 12 | 12 | none |
| spark_greedy_proposals/p2g_ceiling_count_c8_s0 | none | none | none |
| spark_greedy_proposals/p1g_prose_c8_s7 | 19 | 102 | none |
| spark_greedy_proposals/p1g_coding_c2_s0 | 113 | none | none |
| spark_warmup/narrative | 1 | 16 | none |

The retained captures and their immutable manifests are external benchmark inputs; no public corpus download is bundled.
