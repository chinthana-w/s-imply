# Android USB camera notes
- Android 14 QPR1+ can expose compatible phones as USB webcams.
- macOS should see that as a camera device.
- Camera index may vary; run `digit9 cameras` first.
- Use `digit9 live --camera-index N`.
- scrcpy webcam/V4L2 is Linux-only; do not assume on macOS.
- If no UVC mode, use prerecorded video or virtual camera app for prototyping.
