## Codec x bitrate (8 s clip, 720x1280 @ 30 fps, ground truth 9 crossings)

| codec | bitrate kbit/s | count | precision | recall | tracks |
|---|---|---|---|---|---|
| h264 | 3779 | 9 | 1.0 | 1.0 | 38 |
| h264 | 944 | 9 | 1.0 | 1.0 | 36 |
| h264 | 474 | 9 | 1.0 | 1.0 | 36 |
| h264 | 241 | 8 | 1.0 | 0.889 | 34 |
| h264 | 118 | 4 | 1.0 | 0.444 | 56 |
| h265 | 3928 | 9 | 1.0 | 1.0 | 40 |
| h265 | 989 | 9 | 1.0 | 1.0 | 40 |
| h265 | 498 | 9 | 1.0 | 1.0 | 39 |
| h265 | 253 | 7 | 1.0 | 0.778 | 42 |
| h265 | 125 | 4 | 1.0 | 0.444 | 29 |

## Processed FPS (same clip)

| processed fps | count | precision | recall | tracks |
|---|---|---|---|---|
| 30.0 | 9 | 1.0 | 1.0 | 36 |
| 15.0 | 8 | 1.0 | 0.889 | 36 |
| 10.0 | 8 | 1.0 | 0.889 | 33 |
| 7.5 | 6 | 0.833 | 0.556 | 31 |
| 5.0 | 3 | 0.667 | 0.222 | 23 |

## Decode cost (single core, ~1 Mbit/s, 720x1280)

| codec | ms per frame |
|---|---|
| h264 | 1.69 |
| h265 | 2.32 |

## Factory video (464x832 @ 60 fps, 3.9 min): agreement with the full-quality run, not accuracy

| variant | count | vs reference |
|---|---|---|
| original, 30 fps processed | 104 | reference |
| original, 15 fps processed | 104 | 1.0 |
| original, 10 fps processed | 104 | 1.0 |
| original, 5 fps processed | 104 | 1.0 |
| h264 2006 kbit/s, 30 fps processed | 104 | 1.0 |
| h264 506 kbit/s, 30 fps processed | 105 | 1.01 |
| h264 206 kbit/s, 30 fps processed | 95 | 0.913 |
| h265 2008 kbit/s, 30 fps processed | 104 | 1.0 |
| h265 508 kbit/s, 30 fps processed | 104 | 1.0 |
| h265 208 kbit/s, 30 fps processed | 104 | 1.0 |
