# Play Subway Surfers with your eyes

Look left or right to change lanes, wink with the right eye to jump and with the left eye to roll. It uses a normal webcam.

<!-- Add a demo video here -->

## How it works
- **Gaze direction that holds up when the head moves:** instead of the absolute pupil position, I compute the pupil's position *relative to the eye corners* (`(pupil.x - outer_corner.x) / eye_width`). Moving your head moves the corners and the pupil together, so the ratio stays stable.
- **Smoothing:** a 7-frame moving average on the gaze ratio.
- **Decision:** ratio < 0.42 − margin → left arrow, > 0.58 + margin → right arrow, with a 0.4 s cooldown so one glance gives one lane change.
- **Winks:** one eye closed (eyelid distance below a threshold) while the other is open → up or down arrow, with a 0.6 s cooldown. Both eyes closed (natural blink) does nothing.

## Files
- `eye_subway.py`: Python, OpenCV, MediaPipe, `keyboard`
