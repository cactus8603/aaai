# Positive controls (analyze_controls.py)

hard acc = share of items whose twin-averaged target-minus-other logit is > 0 (eligibility metric). mean P is shown for reference only.

## dev_controls (11605 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| easy_long | color/image | 64 | 0.609 | 0.586 | 0 | 0 | 0 |
| easy_long | color/text | 64 | 0.500 | 0.522 | 0 | 0 | 0 |
| easy_long | medical/image | 64 | 0.547 | 0.538 | 0 | 0 | 0 |
| easy_long | medical/text | 64 | 0.578 | 0.575 | 0 | 0 | 0 |
| easy_long | news/image | 64 | 0.484 | 0.482 | 0 | 0 | 0 |
| easy_long | news/text | 64 | 0.531 | 0.514 | 0 | 0 | 0 |
| easy_long | pet/image | 64 | 0.531 | 0.559 | 0 | 0 | 0 |
| easy_long | pet/text | 64 | 0.531 | 0.558 | 0 | 0 | 0 |
| easy_long | sports/image | 64 | 0.516 | 0.520 | 0 | 0 | 0 |
| easy_long | sports/text | 64 | 0.562 | 0.554 | 0 | 0 | 0 |
| easy_long | stock/image | 64 | 0.453 | 0.502 | 0 | 0 | 0 |
| easy_long | stock/text | 64 | 0.484 | 0.509 | 0 | 0 | 0 |
| easy_long | weather/image | 64 | 0.594 | 0.579 | 0 | 0 | 0 |
| easy_long | weather/text | 64 | 0.531 | 0.513 | 0 | 0 | 0 |
| easy_long_coded | color/image | 64 | 0.594 | 0.559 | 0 | 0 | 0 |
| easy_long_coded | color/text | 64 | 0.562 | 0.571 | 0 | 0 | 0 |
| easy_long_coded | medical/image | 64 | 0.625 | 0.571 | 0 | 0 | 0 |
| easy_long_coded | medical/text | 64 | 0.641 | 0.627 | 0 | 0 | 0 |
| easy_long_coded | news/image | 64 | 0.562 | 0.578 | 0 | 0 | 0 |
| easy_long_coded | news/text | 64 | 0.656 | 0.624 | 0 | 0 | 0 |
| easy_long_coded | pet/image | 64 | 0.578 | 0.569 | 0 | 0 | 0 |
| easy_long_coded | pet/text | 64 | 0.641 | 0.647 | 0 | 0 | 0 |
| easy_long_coded | sports/image | 64 | 0.469 | 0.479 | 0 | 0 | 0 |
| easy_long_coded | sports/text | 64 | 0.703 | 0.633 | 0 | 0 | 0 |
| easy_long_coded | stock/image | 64 | 0.656 | 0.629 | 0 | 0 | 0 |
| easy_long_coded | stock/text | 64 | 0.766 | 0.696 | 0 | 0 | 0 |
| easy_long_coded | weather/image | 64 | 0.609 | 0.610 | 0 | 0 | 0 |
| easy_long_coded | weather/text | 64 | 0.766 | 0.677 | 0 | 0 | 0 |
| easy_short | color/image | 64 | 0.719 | 0.663 | 0 | 0 | 0 |
| easy_short | color/text | 64 | 0.812 | 0.770 | 0 | 0 | 0 |
| easy_short | medical/image | 64 | 0.828 | 0.779 | 0 | 0 | 0 |
| easy_short | medical/text | 64 | 0.891 | 0.841 | 0 | 0 | 0 |
| easy_short | news/image | 64 | 0.734 | 0.687 | 0 | 0 | 0 |
| easy_short | news/text | 64 | 0.672 | 0.648 | 0 | 0 | 0 |
| easy_short | pet/image | 64 | 0.609 | 0.553 | 0 | 0 | 0 |
| easy_short | pet/text | 64 | 0.750 | 0.708 | 0 | 0 | 0 |
| easy_short | sports/image | 64 | 0.484 | 0.507 | 0 | 0 | 0 |
| easy_short | sports/text | 64 | 0.641 | 0.635 | 0 | 0 | 0 |
| easy_short | stock/image | 64 | 0.641 | 0.597 | 0 | 0 | 0 |
| easy_short | stock/text | 64 | 0.578 | 0.578 | 0 | 0 | 0 |
| easy_short | weather/image | 64 | 0.469 | 0.519 | 0 | 0 | 0 |
| easy_short | weather/text | 64 | 0.578 | 0.585 | 0 | 0 | 0 |
| easy_short_coded | color/image | 64 | 1.000 | 0.972 | 0 | 0 | 0 |
| easy_short_coded | color/text | 64 | 1.000 | 0.993 | 0 | 0 | 0 |
| easy_short_coded | medical/image | 64 | 1.000 | 0.853 | 0 | 0 | 0 |
| easy_short_coded | medical/text | 64 | 1.000 | 0.943 | 0 | 0 | 0 |
| easy_short_coded | news/image | 64 | 1.000 | 0.995 | 0 | 0 | 0 |
| easy_short_coded | news/text | 64 | 1.000 | 0.976 | 0 | 0 | 0 |
| easy_short_coded | pet/image | 64 | 0.844 | 0.788 | 0 | 0 | 0 |
| easy_short_coded | pet/text | 64 | 1.000 | 0.998 | 0 | 0 | 0 |
| easy_short_coded | sports/image | 64 | 0.797 | 0.785 | 0 | 0 | 0 |
| easy_short_coded | sports/text | 64 | 1.000 | 0.987 | 0 | 0 | 0 |
| easy_short_coded | stock/image | 64 | 1.000 | 0.976 | 0 | 0 | 0 |
| easy_short_coded | stock/text | 64 | 1.000 | 0.998 | 0 | 0 | 0 |
| easy_short_coded | weather/image | 64 | 1.000 | 0.923 | 0 | 0 | 0 |
| easy_short_coded | weather/text | 64 | 1.000 | 0.984 | 0 | 0 | 0 |
| explicit_counts | color/image | 64 | 1.000 | 0.936 | 0 | 0 | 0 |
| explicit_counts | color/text | 64 | 1.000 | 0.967 | 0 | 0 | 0 |
| explicit_counts | medical/image | 64 | 0.953 | 0.808 | 0 | 0 | 0 |
| explicit_counts | medical/text | 64 | 1.000 | 0.879 | 0 | 0 | 0 |
| explicit_counts | news/image | 64 | 0.984 | 0.889 | 0 | 0 | 0 |
| explicit_counts | news/text | 64 | 1.000 | 0.914 | 0 | 0 | 0 |
| explicit_counts | pet/image | 64 | 0.922 | 0.858 | 0 | 0 | 0 |
| explicit_counts | pet/text | 64 | 1.000 | 0.981 | 0 | 0 | 0 |
| explicit_counts | sports/image | 64 | 0.656 | 0.665 | 0 | 0 | 0 |
| explicit_counts | sports/text | 64 | 1.000 | 0.969 | 0 | 0 | 0 |
| explicit_counts | stock/image | 64 | 0.938 | 0.908 | 0 | 0 | 0 |
| explicit_counts | stock/text | 64 | 1.000 | 0.970 | 0 | 0 | 0 |
| explicit_counts | weather/image | 64 | 0.609 | 0.688 | 0 | 0 | 0 |
| explicit_counts | weather/text | 64 | 0.984 | 0.911 | 0 | 0 | 0 |
| explicit_rel | color/image | 64 | 0.875 | 0.838 | 0 | 0 | 0 |
| explicit_rel | color/text | 64 | 1.000 | 0.946 | 0 | 0 | 0 |
| explicit_rel | medical/image | 64 | 0.719 | 0.742 | 0 | 0 | 0 |
| explicit_rel | medical/text | 64 | 0.875 | 0.792 | 0 | 0 | 0 |
| explicit_rel | news/image | 64 | 0.984 | 0.890 | 0 | 0 | 0 |
| explicit_rel | news/text | 64 | 1.000 | 0.912 | 0 | 0 | 0 |
| explicit_rel | pet/image | 64 | 0.875 | 0.797 | 0 | 0 | 0 |
| explicit_rel | pet/text | 64 | 1.000 | 0.968 | 0 | 0 | 0 |
| explicit_rel | sports/image | 64 | 0.844 | 0.731 | 0 | 0 | 0 |
| explicit_rel | sports/text | 64 | 0.906 | 0.824 | 0 | 0 | 0 |
| explicit_rel | stock/image | 64 | 1.000 | 0.874 | 0 | 0 | 0 |
| explicit_rel | stock/text | 64 | 0.859 | 0.849 | 0 | 0 | 0 |
| explicit_rel | weather/image | 64 | 0.609 | 0.617 | 0 | 0 | 0 |
| explicit_rel | weather/text | 64 | 0.797 | 0.770 | 0 | 0 | 0 |
| perception | color | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | medical | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | news | 2 | 1.000 | 0.987 | 0 | 0 | 0 |
| perception | pet | 73 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | sports | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | stock | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | weather | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | medical/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | news/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | sports/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | stock/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | weather/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | nan | not run |
| explicit_rel | 0.8 | 0.882 | pass |
| explicit_rel by domain/format | | | color/image: pass, color/text: pass, medical/image: fail, medical/text: pass, news/image: pass, news/text: pass, pet/image: pass, pet/text: pass, sports/image: pass, sports/text: pass, stock/image: pass, stock/text: pass, weather/image: fail, weather/text: fail |

## dev_binding (2944 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| composite_read | panel1 | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| composite_read | panel2 | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | color/image_after | 64 | 1.000 | 0.850 | 0 | 0 | 0 |
| explicit_rel | color/image_composite | 64 | 1.000 | 0.862 | 0 | 0 | 0 |
| explicit_rel | color/image_named | 64 | 0.891 | 0.823 | 0 | 0 | 0 |
| explicit_rel | medical/image_after | 64 | 0.609 | 0.634 | 0 | 0 | 0 |
| explicit_rel | medical/image_composite | 64 | 0.656 | 0.656 | 0 | 0 | 0 |
| explicit_rel | medical/image_named | 64 | 0.641 | 0.675 | 0 | 0 | 0 |
| explicit_rel | news/image_after | 64 | 0.438 | 0.564 | 0 | 0 | 0 |
| explicit_rel | news/image_composite | 64 | 0.828 | 0.728 | 0 | 0 | 0 |
| explicit_rel | news/image_named | 64 | 0.969 | 0.905 | 0 | 0 | 0 |
| explicit_rel | pet/image_after | 64 | 0.734 | 0.712 | 0 | 0 | 0 |
| explicit_rel | pet/image_composite | 64 | 0.609 | 0.635 | 0 | 0 | 0 |
| explicit_rel | pet/image_named | 64 | 0.875 | 0.830 | 0 | 0 | 0 |
| explicit_rel | sports/image_after | 64 | 0.703 | 0.590 | 0 | 0 | 0 |
| explicit_rel | sports/image_composite | 64 | 0.719 | 0.645 | 0 | 0 | 0 |
| explicit_rel | sports/image_named | 64 | 0.844 | 0.737 | 0 | 0 | 0 |
| explicit_rel | stock/image_after | 64 | 0.438 | 0.440 | 0 | 0 | 0 |
| explicit_rel | stock/image_composite | 64 | 0.719 | 0.721 | 0 | 0 | 0 |
| explicit_rel | stock/image_named | 64 | 1.000 | 0.930 | 0 | 0 | 0 |
| explicit_rel | weather/image_after | 64 | 0.391 | 0.398 | 0 | 0 | 0 |
| explicit_rel | weather/image_composite | 64 | 0.719 | 0.672 | 0 | 0 | 0 |
| explicit_rel | weather/image_named | 64 | 0.609 | 0.621 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | nan | not run |
| explicit_rel | 0.8 | 0.733 | fail |
| explicit_rel by domain/format | | | color/image_after: pass, color/image_composite: pass, color/image_named: pass, medical/image_after: fail, medical/image_composite: fail, medical/image_named: fail, news/image_after: fail, news/image_composite: pass, news/image_named: pass, pet/image_after: fail, pet/image_composite: fail, pet/image_named: pass, sports/image_after: fail, sports/image_composite: fail, sports/image_named: pass, stock/image_after: fail, stock/image_composite: fail, stock/image_named: pass, weather/image_after: fail, weather/image_composite: fail, weather/image_named: fail |

## dev_chartread (384 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| chart_read | all | 192 | 0.620 | 0.596 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | 0.620 | fail |
| explicit_rel | 0.8 | nan | not run |

## dev_controls (11605 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| easy_long | color/image | 64 | 0.688 | 0.602 | 0 | 0 | 0 |
| easy_long | color/text | 64 | 0.609 | 0.546 | 0 | 0 | 0 |
| easy_long | medical/image | 64 | 0.500 | 0.481 | 0 | 0 | 0 |
| easy_long | medical/text | 64 | 0.641 | 0.584 | 0 | 0 | 0 |
| easy_long | news/image | 64 | 0.422 | 0.483 | 0 | 0 | 0 |
| easy_long | news/text | 64 | 0.578 | 0.525 | 0 | 0 | 0 |
| easy_long | pet/image | 64 | 0.500 | 0.491 | 0 | 0 | 0 |
| easy_long | pet/text | 64 | 0.562 | 0.511 | 0 | 0 | 0 |
| easy_long | sports/image | 64 | 0.516 | 0.502 | 0 | 0 | 0 |
| easy_long | sports/text | 64 | 0.609 | 0.551 | 0 | 0 | 0 |
| easy_long | stock/image | 64 | 0.531 | 0.487 | 0 | 0 | 0 |
| easy_long | stock/text | 64 | 0.422 | 0.497 | 0 | 0 | 0 |
| easy_long | weather/image | 64 | 0.484 | 0.497 | 0 | 0 | 0 |
| easy_long | weather/text | 64 | 0.547 | 0.507 | 0 | 0 | 0 |
| easy_long_coded | color/image | 64 | 0.391 | 0.448 | 0 | 0 | 0 |
| easy_long_coded | color/text | 64 | 0.625 | 0.615 | 0 | 0 | 0 |
| easy_long_coded | medical/image | 64 | 0.406 | 0.486 | 0 | 0 | 0 |
| easy_long_coded | medical/text | 64 | 0.594 | 0.571 | 0 | 0 | 0 |
| easy_long_coded | news/image | 64 | 0.547 | 0.518 | 0 | 0 | 0 |
| easy_long_coded | news/text | 64 | 0.641 | 0.586 | 0 | 0 | 0 |
| easy_long_coded | pet/image | 64 | 0.406 | 0.435 | 0 | 0 | 0 |
| easy_long_coded | pet/text | 64 | 0.703 | 0.667 | 0 | 0 | 0 |
| easy_long_coded | sports/image | 64 | 0.484 | 0.474 | 0 | 0 | 0 |
| easy_long_coded | sports/text | 64 | 0.750 | 0.673 | 0 | 0 | 0 |
| easy_long_coded | stock/image | 64 | 0.422 | 0.455 | 0 | 0 | 0 |
| easy_long_coded | stock/text | 64 | 0.734 | 0.644 | 0 | 0 | 0 |
| easy_long_coded | weather/image | 64 | 0.484 | 0.488 | 0 | 0 | 0 |
| easy_long_coded | weather/text | 64 | 0.672 | 0.632 | 0 | 0 | 0 |
| easy_short | color/image | 64 | 0.469 | 0.523 | 0 | 0 | 0 |
| easy_short | color/text | 64 | 0.641 | 0.625 | 0 | 0 | 0 |
| easy_short | medical/image | 64 | 0.625 | 0.567 | 0 | 0 | 0 |
| easy_short | medical/text | 64 | 0.828 | 0.841 | 0 | 0 | 0 |
| easy_short | news/image | 64 | 0.703 | 0.610 | 0 | 0 | 0 |
| easy_short | news/text | 64 | 0.797 | 0.664 | 0 | 0 | 0 |
| easy_short | pet/image | 64 | 0.453 | 0.456 | 0 | 0 | 0 |
| easy_short | pet/text | 64 | 0.500 | 0.515 | 0 | 0 | 0 |
| easy_short | sports/image | 64 | 0.516 | 0.548 | 0 | 0 | 0 |
| easy_short | sports/text | 64 | 0.703 | 0.626 | 0 | 0 | 0 |
| easy_short | stock/image | 64 | 0.531 | 0.518 | 0 | 0 | 0 |
| easy_short | stock/text | 64 | 0.453 | 0.511 | 0 | 0 | 0 |
| easy_short | weather/image | 64 | 0.500 | 0.472 | 0 | 0 | 0 |
| easy_short | weather/text | 64 | 0.500 | 0.544 | 0 | 0 | 0 |
| easy_short_coded | color/image | 64 | 0.734 | 0.748 | 0 | 0 | 0 |
| easy_short_coded | color/text | 64 | 1.000 | 0.987 | 0 | 0 | 0 |
| easy_short_coded | medical/image | 64 | 0.516 | 0.558 | 0 | 0 | 0 |
| easy_short_coded | medical/text | 64 | 0.969 | 0.862 | 0 | 0 | 0 |
| easy_short_coded | news/image | 64 | 0.938 | 0.773 | 0 | 0 | 0 |
| easy_short_coded | news/text | 64 | 1.000 | 0.929 | 0 | 0 | 0 |
| easy_short_coded | pet/image | 64 | 0.781 | 0.734 | 0 | 0 | 0 |
| easy_short_coded | pet/text | 64 | 1.000 | 0.994 | 0 | 0 | 0 |
| easy_short_coded | sports/image | 64 | 1.000 | 0.873 | 0 | 0 | 0 |
| easy_short_coded | sports/text | 64 | 1.000 | 0.960 | 0 | 0 | 0 |
| easy_short_coded | stock/image | 64 | 0.625 | 0.684 | 0 | 0 | 0 |
| easy_short_coded | stock/text | 64 | 1.000 | 0.976 | 0 | 0 | 0 |
| easy_short_coded | weather/image | 64 | 0.609 | 0.577 | 0 | 0 | 0 |
| easy_short_coded | weather/text | 64 | 1.000 | 0.982 | 0 | 0 | 0 |
| explicit_counts | color/image | 64 | 0.938 | 0.872 | 0 | 0 | 0 |
| explicit_counts | color/text | 64 | 1.000 | 0.916 | 0 | 0 | 0 |
| explicit_counts | medical/image | 64 | 0.516 | 0.536 | 0 | 0 | 0 |
| explicit_counts | medical/text | 64 | 1.000 | 0.901 | 0 | 0 | 0 |
| explicit_counts | news/image | 64 | 0.891 | 0.694 | 0 | 0 | 0 |
| explicit_counts | news/text | 64 | 1.000 | 0.878 | 0 | 0 | 0 |
| explicit_counts | pet/image | 64 | 0.938 | 0.857 | 0 | 0 | 0 |
| explicit_counts | pet/text | 64 | 1.000 | 0.892 | 0 | 0 | 0 |
| explicit_counts | sports/image | 64 | 0.734 | 0.734 | 0 | 0 | 0 |
| explicit_counts | sports/text | 64 | 1.000 | 0.858 | 0 | 0 | 0 |
| explicit_counts | stock/image | 64 | 0.766 | 0.691 | 0 | 0 | 0 |
| explicit_counts | stock/text | 64 | 1.000 | 0.890 | 0 | 0 | 0 |
| explicit_counts | weather/image | 64 | 0.719 | 0.561 | 0 | 0 | 0 |
| explicit_counts | weather/text | 64 | 1.000 | 0.944 | 0 | 0 | 0 |
| explicit_rel | color/image | 64 | 0.859 | 0.828 | 0 | 0 | 0 |
| explicit_rel | color/text | 64 | 1.000 | 0.940 | 0 | 0 | 0 |
| explicit_rel | medical/image | 64 | 0.688 | 0.558 | 0 | 0 | 0 |
| explicit_rel | medical/text | 64 | 1.000 | 0.900 | 0 | 0 | 0 |
| explicit_rel | news/image | 64 | 0.750 | 0.717 | 0 | 0 | 0 |
| explicit_rel | news/text | 64 | 1.000 | 0.916 | 0 | 0 | 0 |
| explicit_rel | pet/image | 64 | 0.844 | 0.775 | 0 | 0 | 0 |
| explicit_rel | pet/text | 64 | 1.000 | 0.891 | 0 | 0 | 0 |
| explicit_rel | sports/image | 64 | 0.781 | 0.754 | 0 | 0 | 0 |
| explicit_rel | sports/text | 64 | 1.000 | 0.889 | 0 | 0 | 0 |
| explicit_rel | stock/image | 64 | 0.953 | 0.812 | 0 | 0 | 0 |
| explicit_rel | stock/text | 64 | 1.000 | 0.950 | 0 | 0 | 0 |
| explicit_rel | weather/image | 64 | 0.719 | 0.565 | 0 | 0 | 0 |
| explicit_rel | weather/text | 64 | 1.000 | 0.955 | 0 | 0 | 0 |
| perception | color | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | medical | 2 | 1.000 | 0.851 | 0 | 0 | 0 |
| perception | news | 2 | 1.000 | 0.997 | 0 | 0 | 0 |
| perception | pet | 73 | 0.986 | 0.989 | 0 | 0 | 0 |
| perception | sports | 2 | 1.000 | 0.987 | 0 | 0 | 0 |
| perception | stock | 2 | 1.000 | 0.998 | 0 | 0 | 0 |
| perception | weather | 2 | 1.000 | 0.882 | 0 | 0 | 0 |
| single_trial_who | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | medical/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | news/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | sports/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | stock/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | weather/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | nan | not run |
| explicit_rel | 0.8 | 0.900 | pass |
| explicit_rel by domain/format | | | color/image: pass, color/text: pass, medical/image: fail, medical/text: pass, news/image: fail, news/text: pass, pet/image: pass, pet/text: pass, sports/image: fail, sports/text: pass, stock/image: pass, stock/text: pass, weather/image: fail, weather/text: pass |

## dev_binding (2944 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| composite_read | panel1 | 64 | 1.000 | 0.999 | 0 | 0 | 0 |
| composite_read | panel2 | 64 | 1.000 | 0.998 | 0 | 0 | 0 |
| explicit_rel | color/image_after | 64 | 0.844 | 0.771 | 0 | 0 | 0 |
| explicit_rel | color/image_composite | 64 | 1.000 | 0.970 | 0 | 0 | 0 |
| explicit_rel | color/image_named | 64 | 1.000 | 0.979 | 0 | 0 | 0 |
| explicit_rel | medical/image_after | 64 | 0.812 | 0.599 | 0 | 0 | 0 |
| explicit_rel | medical/image_composite | 64 | 0.750 | 0.655 | 0 | 0 | 0 |
| explicit_rel | medical/image_named | 64 | 0.594 | 0.523 | 0 | 0 | 0 |
| explicit_rel | news/image_after | 64 | 0.594 | 0.602 | 0 | 0 | 0 |
| explicit_rel | news/image_composite | 64 | 0.844 | 0.709 | 0 | 0 | 0 |
| explicit_rel | news/image_named | 64 | 0.672 | 0.680 | 0 | 0 | 0 |
| explicit_rel | pet/image_after | 64 | 0.625 | 0.622 | 0 | 0 | 0 |
| explicit_rel | pet/image_composite | 64 | 0.766 | 0.744 | 0 | 0 | 0 |
| explicit_rel | pet/image_named | 64 | 0.906 | 0.873 | 0 | 0 | 0 |
| explicit_rel | sports/image_after | 64 | 0.797 | 0.733 | 0 | 0 | 0 |
| explicit_rel | sports/image_composite | 64 | 0.844 | 0.810 | 0 | 0 | 0 |
| explicit_rel | sports/image_named | 64 | 0.844 | 0.777 | 0 | 0 | 0 |
| explicit_rel | stock/image_after | 64 | 0.500 | 0.534 | 0 | 0 | 0 |
| explicit_rel | stock/image_composite | 64 | 0.984 | 0.912 | 0 | 0 | 0 |
| explicit_rel | stock/image_named | 64 | 0.891 | 0.792 | 0 | 0 | 0 |
| explicit_rel | weather/image_after | 64 | 0.719 | 0.540 | 0 | 0 | 0 |
| explicit_rel | weather/image_composite | 64 | 0.781 | 0.766 | 0 | 0 | 0 |
| explicit_rel | weather/image_named | 64 | 0.922 | 0.763 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | nan | not run |
| explicit_rel | 0.8 | 0.795 | fail |
| explicit_rel by domain/format | | | color/image_after: pass, color/image_composite: pass, color/image_named: pass, medical/image_after: pass, medical/image_composite: fail, medical/image_named: fail, news/image_after: fail, news/image_composite: pass, news/image_named: fail, pet/image_after: fail, pet/image_composite: fail, pet/image_named: pass, sports/image_after: fail, sports/image_composite: pass, sports/image_named: pass, stock/image_after: fail, stock/image_composite: pass, stock/image_named: pass, weather/image_after: fail, weather/image_composite: fail, weather/image_named: pass |

## dev_chartread (384 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| chart_read | all | 192 | 0.688 | 0.645 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | 0.688 | fail |
| explicit_rel | 0.8 | nan | not run |

## dev_controls (11605 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| easy_long | color/image | 64 | 0.422 | 0.457 | 0 | 0 | 0 |
| easy_long | color/text | 64 | 0.562 | 0.516 | 0 | 0 | 0 |
| easy_long | medical/image | 64 | 0.531 | 0.498 | 0 | 0 | 0 |
| easy_long | medical/text | 64 | 0.609 | 0.572 | 0 | 0 | 0 |
| easy_long | news/image | 64 | 0.469 | 0.486 | 0 | 0 | 0 |
| easy_long | news/text | 64 | 0.562 | 0.520 | 0 | 0 | 0 |
| easy_long | pet/image | 64 | 0.391 | 0.391 | 0 | 0 | 0 |
| easy_long | pet/text | 64 | 0.422 | 0.459 | 0 | 0 | 0 |
| easy_long | sports/image | 64 | 0.375 | 0.448 | 0 | 0 | 0 |
| easy_long | sports/text | 64 | 0.594 | 0.557 | 0 | 0 | 0 |
| easy_long | stock/image | 64 | 0.391 | 0.387 | 0 | 0 | 0 |
| easy_long | stock/text | 64 | 0.516 | 0.505 | 0 | 0 | 0 |
| easy_long | weather/image | 64 | 0.391 | 0.386 | 0 | 0 | 0 |
| easy_long | weather/text | 64 | 0.625 | 0.559 | 0 | 0 | 0 |
| easy_long_coded | color/image | 64 | 0.797 | 0.704 | 0 | 0 | 0 |
| easy_long_coded | color/text | 64 | 0.859 | 0.786 | 0 | 0 | 0 |
| easy_long_coded | medical/image | 64 | 0.547 | 0.536 | 0 | 0 | 0 |
| easy_long_coded | medical/text | 64 | 0.703 | 0.608 | 0 | 0 | 0 |
| easy_long_coded | news/image | 64 | 0.516 | 0.521 | 0 | 0 | 0 |
| easy_long_coded | news/text | 64 | 0.656 | 0.631 | 0 | 0 | 0 |
| easy_long_coded | pet/image | 64 | 0.391 | 0.408 | 0 | 0 | 0 |
| easy_long_coded | pet/text | 64 | 0.812 | 0.737 | 0 | 0 | 0 |
| easy_long_coded | sports/image | 64 | 0.422 | 0.464 | 0 | 0 | 0 |
| easy_long_coded | sports/text | 64 | 0.688 | 0.612 | 0 | 0 | 0 |
| easy_long_coded | stock/image | 64 | 0.391 | 0.423 | 0 | 0 | 0 |
| easy_long_coded | stock/text | 64 | 0.844 | 0.709 | 0 | 0 | 0 |
| easy_long_coded | weather/image | 64 | 0.391 | 0.446 | 0 | 0 | 0 |
| easy_long_coded | weather/text | 64 | 0.750 | 0.648 | 0 | 0 | 0 |
| easy_short | color/image | 64 | 0.656 | 0.633 | 0 | 0 | 0 |
| easy_short | color/text | 64 | 0.719 | 0.695 | 0 | 0 | 0 |
| easy_short | medical/image | 64 | 0.641 | 0.640 | 0 | 0 | 0 |
| easy_short | medical/text | 64 | 0.906 | 0.888 | 0 | 0 | 0 |
| easy_short | news/image | 64 | 0.688 | 0.689 | 0 | 0 | 0 |
| easy_short | news/text | 64 | 0.547 | 0.565 | 0 | 0 | 0 |
| easy_short | pet/image | 64 | 0.391 | 0.390 | 0 | 0 | 0 |
| easy_short | pet/text | 64 | 0.500 | 0.531 | 0 | 0 | 0 |
| easy_short | sports/image | 64 | 0.422 | 0.463 | 0 | 0 | 0 |
| easy_short | sports/text | 64 | 0.516 | 0.548 | 0 | 0 | 0 |
| easy_short | stock/image | 64 | 0.375 | 0.407 | 0 | 0 | 0 |
| easy_short | stock/text | 64 | 0.531 | 0.522 | 0 | 0 | 0 |
| easy_short | weather/image | 64 | 0.453 | 0.448 | 0 | 0 | 0 |
| easy_short | weather/text | 64 | 0.531 | 0.553 | 0 | 0 | 0 |
| easy_short_coded | color/image | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | medical/image | 64 | 0.641 | 0.700 | 0 | 0 | 0 |
| easy_short_coded | medical/text | 64 | 1.000 | 0.998 | 0 | 0 | 0 |
| easy_short_coded | news/image | 64 | 0.906 | 0.858 | 0 | 0 | 0 |
| easy_short_coded | news/text | 64 | 1.000 | 0.962 | 0 | 0 | 0 |
| easy_short_coded | pet/image | 64 | 0.391 | 0.412 | 0 | 0 | 0 |
| easy_short_coded | pet/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | sports/image | 64 | 0.641 | 0.633 | 0 | 0 | 0 |
| easy_short_coded | sports/text | 64 | 1.000 | 0.991 | 0 | 0 | 0 |
| easy_short_coded | stock/image | 64 | 0.625 | 0.603 | 0 | 0 | 0 |
| easy_short_coded | stock/text | 64 | 1.000 | 0.999 | 0 | 0 | 0 |
| easy_short_coded | weather/image | 64 | 0.703 | 0.700 | 0 | 0 | 0 |
| easy_short_coded | weather/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_counts | color/image | 64 | 1.000 | 0.999 | 0 | 0 | 0 |
| explicit_counts | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_counts | medical/image | 64 | 0.703 | 0.703 | 0 | 0 | 0 |
| explicit_counts | medical/text | 64 | 1.000 | 0.999 | 0 | 0 | 0 |
| explicit_counts | news/image | 64 | 0.641 | 0.624 | 0 | 0 | 0 |
| explicit_counts | news/text | 64 | 1.000 | 0.979 | 0 | 0 | 0 |
| explicit_counts | pet/image | 64 | 0.391 | 0.414 | 0 | 0 | 0 |
| explicit_counts | pet/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_counts | sports/image | 64 | 0.641 | 0.607 | 0 | 0 | 0 |
| explicit_counts | sports/text | 64 | 1.000 | 0.989 | 0 | 0 | 0 |
| explicit_counts | stock/image | 64 | 0.625 | 0.615 | 0 | 0 | 0 |
| explicit_counts | stock/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_counts | weather/image | 64 | 0.516 | 0.565 | 0 | 0 | 0 |
| explicit_counts | weather/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | color/image | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | medical/image | 64 | 0.703 | 0.676 | 0 | 0 | 0 |
| explicit_rel | medical/text | 64 | 1.000 | 0.997 | 0 | 0 | 0 |
| explicit_rel | news/image | 64 | 0.641 | 0.658 | 0 | 0 | 0 |
| explicit_rel | news/text | 64 | 1.000 | 0.983 | 0 | 0 | 0 |
| explicit_rel | pet/image | 64 | 0.391 | 0.392 | 0 | 0 | 0 |
| explicit_rel | pet/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | sports/image | 64 | 0.703 | 0.705 | 0 | 0 | 0 |
| explicit_rel | sports/text | 64 | 1.000 | 0.997 | 0 | 0 | 0 |
| explicit_rel | stock/image | 64 | 0.688 | 0.724 | 0 | 0 | 0 |
| explicit_rel | stock/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | weather/image | 64 | 0.656 | 0.673 | 0 | 0 | 0 |
| explicit_rel | weather/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | color | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | medical | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | news | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | pet | 73 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | sports | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | stock | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | weather | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | medical/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | news/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | sports/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | stock/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | weather/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | nan | not run |
| explicit_rel | 0.8 | 0.842 | pass |
| explicit_rel by domain/format | | | color/image: pass, color/text: pass, medical/image: fail, medical/text: pass, news/image: fail, news/text: pass, pet/image: fail, pet/text: pass, sports/image: fail, sports/text: pass, stock/image: fail, stock/text: pass, weather/image: fail, weather/text: pass |

## dev_binding (2944 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| composite_read | panel1 | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| composite_read | panel2 | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | color/image_after | 64 | 0.969 | 0.927 | 0 | 0 | 0 |
| explicit_rel | color/image_composite | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | color/image_named | 64 | 1.000 | 0.995 | 0 | 0 | 0 |
| explicit_rel | medical/image_after | 64 | 0.672 | 0.629 | 0 | 0 | 0 |
| explicit_rel | medical/image_composite | 64 | 0.812 | 0.758 | 0 | 0 | 0 |
| explicit_rel | medical/image_named | 64 | 0.875 | 0.825 | 0 | 0 | 0 |
| explicit_rel | news/image_after | 64 | 0.609 | 0.545 | 0 | 0 | 0 |
| explicit_rel | news/image_composite | 64 | 0.641 | 0.618 | 0 | 0 | 0 |
| explicit_rel | news/image_named | 64 | 0.641 | 0.618 | 0 | 0 | 0 |
| explicit_rel | pet/image_after | 64 | 0.391 | 0.399 | 0 | 0 | 0 |
| explicit_rel | pet/image_composite | 64 | 0.672 | 0.676 | 0 | 0 | 0 |
| explicit_rel | pet/image_named | 64 | 0.422 | 0.447 | 0 | 0 | 0 |
| explicit_rel | sports/image_after | 64 | 0.391 | 0.427 | 0 | 0 | 0 |
| explicit_rel | sports/image_composite | 64 | 0.516 | 0.530 | 0 | 0 | 0 |
| explicit_rel | sports/image_named | 64 | 0.641 | 0.653 | 0 | 0 | 0 |
| explicit_rel | stock/image_after | 64 | 0.438 | 0.456 | 0 | 0 | 0 |
| explicit_rel | stock/image_composite | 64 | 0.719 | 0.739 | 0 | 0 | 0 |
| explicit_rel | stock/image_named | 64 | 0.797 | 0.767 | 0 | 0 | 0 |
| explicit_rel | weather/image_after | 64 | 0.391 | 0.391 | 0 | 0 | 0 |
| explicit_rel | weather/image_composite | 64 | 1.000 | 0.998 | 0 | 0 | 0 |
| explicit_rel | weather/image_named | 64 | 0.953 | 0.930 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | nan | not run |
| explicit_rel | 0.8 | 0.693 | fail |
| explicit_rel by domain/format | | | color/image_after: pass, color/image_composite: pass, color/image_named: pass, medical/image_after: fail, medical/image_composite: pass, medical/image_named: pass, news/image_after: fail, news/image_composite: fail, news/image_named: fail, pet/image_after: fail, pet/image_composite: fail, pet/image_named: fail, sports/image_after: fail, sports/image_composite: fail, sports/image_named: fail, stock/image_after: fail, stock/image_composite: fail, stock/image_named: fail, weather/image_after: fail, weather/image_composite: pass, weather/image_named: pass |

## dev_chartread (384 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| chart_read | all | 192 | 0.672 | 0.682 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | 0.672 | fail |
| explicit_rel | 0.8 | nan | not run |

## dev_controls (11605 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| easy_long | color/image | 64 | 0.547 | 0.530 | 0 | 0 | 0 |
| easy_long | color/text | 64 | 0.766 | 0.650 | 0 | 0 | 0 |
| easy_long | medical/image | 64 | 0.500 | 0.491 | 0 | 0 | 0 |
| easy_long | medical/text | 64 | 0.672 | 0.631 | 0 | 0 | 0 |
| easy_long | news/image | 64 | 0.547 | 0.537 | 0 | 0 | 0 |
| easy_long | news/text | 64 | 0.734 | 0.637 | 0 | 0 | 0 |
| easy_long | pet/image | 64 | 0.406 | 0.404 | 0 | 0 | 0 |
| easy_long | pet/text | 64 | 0.688 | 0.618 | 0 | 0 | 0 |
| easy_long | sports/image | 64 | 0.406 | 0.473 | 0 | 0 | 0 |
| easy_long | sports/text | 64 | 0.750 | 0.624 | 0 | 0 | 0 |
| easy_long | stock/image | 64 | 0.484 | 0.512 | 0 | 0 | 0 |
| easy_long | stock/text | 64 | 0.609 | 0.580 | 0 | 0 | 0 |
| easy_long | weather/image | 64 | 0.344 | 0.472 | 0 | 0 | 0 |
| easy_long | weather/text | 64 | 0.656 | 0.590 | 0 | 0 | 0 |
| easy_long_coded | color/image | 64 | 0.828 | 0.744 | 0 | 0 | 0 |
| easy_long_coded | color/text | 64 | 0.922 | 0.819 | 0 | 0 | 0 |
| easy_long_coded | medical/image | 64 | 0.750 | 0.533 | 0 | 0 | 0 |
| easy_long_coded | medical/text | 64 | 0.891 | 0.705 | 0 | 0 | 0 |
| easy_long_coded | news/image | 64 | 0.391 | 0.442 | 0 | 0 | 0 |
| easy_long_coded | news/text | 64 | 0.906 | 0.800 | 0 | 0 | 0 |
| easy_long_coded | pet/image | 64 | 0.391 | 0.407 | 0 | 0 | 0 |
| easy_long_coded | pet/text | 64 | 0.953 | 0.844 | 0 | 0 | 0 |
| easy_long_coded | sports/image | 64 | 0.391 | 0.423 | 0 | 0 | 0 |
| easy_long_coded | sports/text | 64 | 0.875 | 0.684 | 0 | 0 | 0 |
| easy_long_coded | stock/image | 64 | 0.547 | 0.517 | 0 | 0 | 0 |
| easy_long_coded | stock/text | 64 | 0.844 | 0.714 | 0 | 0 | 0 |
| easy_long_coded | weather/image | 64 | 0.453 | 0.520 | 0 | 0 | 0 |
| easy_long_coded | weather/text | 64 | 0.906 | 0.708 | 0 | 0 | 0 |
| easy_short | color/image | 64 | 0.609 | 0.600 | 0 | 0 | 0 |
| easy_short | color/text | 64 | 0.828 | 0.808 | 0 | 0 | 0 |
| easy_short | medical/image | 64 | 0.609 | 0.578 | 0 | 0 | 0 |
| easy_short | medical/text | 64 | 0.938 | 0.897 | 0 | 0 | 0 |
| easy_short | news/image | 64 | 0.703 | 0.665 | 0 | 0 | 0 |
| easy_short | news/text | 64 | 0.938 | 0.881 | 0 | 0 | 0 |
| easy_short | pet/image | 64 | 0.391 | 0.386 | 0 | 0 | 0 |
| easy_short | pet/text | 64 | 0.734 | 0.714 | 0 | 0 | 0 |
| easy_short | sports/image | 64 | 0.484 | 0.493 | 0 | 0 | 0 |
| easy_short | sports/text | 64 | 0.750 | 0.706 | 0 | 0 | 0 |
| easy_short | stock/image | 64 | 0.516 | 0.515 | 0 | 0 | 0 |
| easy_short | stock/text | 64 | 0.641 | 0.630 | 0 | 0 | 0 |
| easy_short | weather/image | 64 | 0.469 | 0.481 | 0 | 0 | 0 |
| easy_short | weather/text | 64 | 0.672 | 0.674 | 0 | 0 | 0 |
| easy_short_coded | color/image | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | medical/image | 64 | 1.000 | 0.854 | 0 | 0 | 0 |
| easy_short_coded | medical/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | news/image | 64 | 0.422 | 0.589 | 0 | 0 | 0 |
| easy_short_coded | news/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | pet/image | 64 | 0.391 | 0.414 | 0 | 0 | 0 |
| easy_short_coded | pet/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | sports/image | 64 | 0.406 | 0.493 | 0 | 0 | 0 |
| easy_short_coded | sports/text | 64 | 1.000 | 0.997 | 0 | 0 | 0 |
| easy_short_coded | stock/image | 64 | 0.500 | 0.555 | 0 | 0 | 0 |
| easy_short_coded | stock/text | 64 | 1.000 | 0.993 | 0 | 0 | 0 |
| easy_short_coded | weather/image | 64 | 0.500 | 0.556 | 0 | 0 | 0 |
| easy_short_coded | weather/text | 64 | 1.000 | 0.999 | 0 | 0 | 0 |
| explicit_counts | color/image | 64 | 0.969 | 0.955 | 0 | 0 | 0 |
| explicit_counts | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_counts | medical/image | 64 | 0.750 | 0.672 | 0 | 0 | 0 |
| explicit_counts | medical/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_counts | news/image | 64 | 0.391 | 0.441 | 0 | 0 | 0 |
| explicit_counts | news/text | 64 | 1.000 | 0.997 | 0 | 0 | 0 |
| explicit_counts | pet/image | 64 | 0.406 | 0.427 | 0 | 0 | 0 |
| explicit_counts | pet/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_counts | sports/image | 64 | 0.391 | 0.422 | 0 | 0 | 0 |
| explicit_counts | sports/text | 64 | 1.000 | 0.958 | 0 | 0 | 0 |
| explicit_counts | stock/image | 64 | 0.391 | 0.456 | 0 | 0 | 0 |
| explicit_counts | stock/text | 64 | 1.000 | 0.996 | 0 | 0 | 0 |
| explicit_counts | weather/image | 64 | 0.391 | 0.443 | 0 | 0 | 0 |
| explicit_counts | weather/text | 64 | 1.000 | 0.998 | 0 | 0 | 0 |
| explicit_rel | color/image | 64 | 1.000 | 0.996 | 0 | 0 | 0 |
| explicit_rel | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | medical/image | 64 | 0.938 | 0.731 | 0 | 0 | 0 |
| explicit_rel | medical/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | news/image | 64 | 0.547 | 0.591 | 0 | 0 | 0 |
| explicit_rel | news/text | 64 | 1.000 | 0.999 | 0 | 0 | 0 |
| explicit_rel | pet/image | 64 | 0.391 | 0.409 | 0 | 0 | 0 |
| explicit_rel | pet/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | sports/image | 64 | 0.453 | 0.473 | 0 | 0 | 0 |
| explicit_rel | sports/text | 64 | 1.000 | 0.963 | 0 | 0 | 0 |
| explicit_rel | stock/image | 64 | 0.438 | 0.487 | 0 | 0 | 0 |
| explicit_rel | stock/text | 64 | 1.000 | 0.998 | 0 | 0 | 0 |
| explicit_rel | weather/image | 64 | 0.391 | 0.437 | 0 | 0 | 0 |
| explicit_rel | weather/text | 64 | 1.000 | 0.999 | 0 | 0 | 0 |
| perception | color | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | medical | 2 | 1.000 | 0.946 | 0 | 0 | 0 |
| perception | news | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | pet | 73 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | sports | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | stock | 2 | 1.000 | 0.995 | 0 | 0 | 0 |
| perception | weather | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | medical/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | news/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | sports/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | stock/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | weather/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | nan | not run |
| explicit_rel | 0.8 | 0.797 | fail |
| explicit_rel by domain/format | | | color/image: pass, color/text: pass, medical/image: pass, medical/text: pass, news/image: fail, news/text: pass, pet/image: fail, pet/text: pass, sports/image: fail, sports/text: pass, stock/image: fail, stock/text: pass, weather/image: fail, weather/text: pass |

## dev_binding (2944 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| composite_read | panel1 | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| composite_read | panel2 | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | color/image_after | 64 | 1.000 | 0.969 | 0 | 0 | 0 |
| explicit_rel | color/image_composite | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | color/image_named | 64 | 1.000 | 0.999 | 0 | 0 | 0 |
| explicit_rel | medical/image_after | 64 | 1.000 | 0.748 | 0 | 0 | 0 |
| explicit_rel | medical/image_composite | 64 | 1.000 | 0.930 | 0 | 0 | 0 |
| explicit_rel | medical/image_named | 64 | 0.969 | 0.852 | 0 | 0 | 0 |
| explicit_rel | news/image_after | 64 | 0.703 | 0.682 | 0 | 0 | 0 |
| explicit_rel | news/image_composite | 64 | 1.000 | 0.939 | 0 | 0 | 0 |
| explicit_rel | news/image_named | 64 | 0.406 | 0.500 | 0 | 0 | 0 |
| explicit_rel | pet/image_after | 64 | 0.391 | 0.412 | 0 | 0 | 0 |
| explicit_rel | pet/image_composite | 64 | 0.953 | 0.859 | 0 | 0 | 0 |
| explicit_rel | pet/image_named | 64 | 0.438 | 0.503 | 0 | 0 | 0 |
| explicit_rel | sports/image_after | 64 | 0.469 | 0.493 | 0 | 0 | 0 |
| explicit_rel | sports/image_composite | 64 | 0.891 | 0.759 | 0 | 0 | 0 |
| explicit_rel | sports/image_named | 64 | 0.500 | 0.596 | 0 | 0 | 0 |
| explicit_rel | stock/image_after | 64 | 0.578 | 0.555 | 0 | 0 | 0 |
| explicit_rel | stock/image_composite | 64 | 0.938 | 0.646 | 0 | 0 | 0 |
| explicit_rel | stock/image_named | 64 | 0.547 | 0.578 | 0 | 0 | 0 |
| explicit_rel | weather/image_after | 64 | 0.516 | 0.540 | 0 | 0 | 0 |
| explicit_rel | weather/image_composite | 64 | 1.000 | 0.968 | 0 | 0 | 0 |
| explicit_rel | weather/image_named | 64 | 0.547 | 0.603 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | nan | not run |
| explicit_rel | 0.8 | 0.754 | fail |
| explicit_rel by domain/format | | | color/image_after: pass, color/image_composite: pass, color/image_named: pass, medical/image_after: pass, medical/image_composite: pass, medical/image_named: pass, news/image_after: fail, news/image_composite: pass, news/image_named: fail, pet/image_after: fail, pet/image_composite: pass, pet/image_named: fail, sports/image_after: fail, sports/image_composite: pass, sports/image_named: fail, stock/image_after: fail, stock/image_composite: pass, stock/image_named: fail, weather/image_after: fail, weather/image_composite: pass, weather/image_named: fail |

## dev_chartread (384 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| chart_read | all | 192 | 0.708 | 0.658 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | 0.708 | fail |
| explicit_rel | 0.8 | nan | not run |

## dev_controls (11605 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| easy_long | color/image | 64 | 0.641 | 0.636 | 0 | 0 | 0 |
| easy_long | color/text | 64 | 0.594 | 0.578 | 0 | 0 | 0 |
| easy_long | medical/image | 64 | 0.578 | 0.585 | 0 | 0 | 0 |
| easy_long | medical/text | 64 | 0.562 | 0.556 | 0 | 0 | 0 |
| easy_long | news/image | 64 | 0.656 | 0.618 | 0 | 0 | 0 |
| easy_long | news/text | 64 | 0.656 | 0.634 | 0 | 0 | 0 |
| easy_long | pet/image | 64 | 0.375 | 0.382 | 0 | 0 | 0 |
| easy_long | pet/text | 64 | 0.484 | 0.511 | 0 | 0 | 0 |
| easy_long | sports/image | 64 | 0.531 | 0.505 | 0 | 0 | 0 |
| easy_long | sports/text | 64 | 0.562 | 0.562 | 0 | 0 | 0 |
| easy_long | stock/image | 64 | 0.453 | 0.459 | 0 | 0 | 0 |
| easy_long | stock/text | 64 | 0.469 | 0.492 | 0 | 0 | 0 |
| easy_long | weather/image | 64 | 0.375 | 0.392 | 0 | 0 | 0 |
| easy_long | weather/text | 64 | 0.484 | 0.497 | 0 | 0 | 0 |
| easy_long_coded | color/image | 64 | 0.750 | 0.749 | 0 | 0 | 0 |
| easy_long_coded | color/text | 64 | 0.953 | 0.952 | 0 | 0 | 0 |
| easy_long_coded | medical/image | 64 | 0.438 | 0.462 | 0 | 0 | 0 |
| easy_long_coded | medical/text | 64 | 0.828 | 0.775 | 0 | 0 | 0 |
| easy_long_coded | news/image | 64 | 0.812 | 0.741 | 0 | 0 | 0 |
| easy_long_coded | news/text | 64 | 0.812 | 0.756 | 0 | 0 | 0 |
| easy_long_coded | pet/image | 64 | 0.500 | 0.504 | 0 | 0 | 0 |
| easy_long_coded | pet/text | 64 | 0.984 | 0.959 | 0 | 0 | 0 |
| easy_long_coded | sports/image | 64 | 0.656 | 0.668 | 0 | 0 | 0 |
| easy_long_coded | sports/text | 64 | 0.734 | 0.710 | 0 | 0 | 0 |
| easy_long_coded | stock/image | 64 | 0.547 | 0.552 | 0 | 0 | 0 |
| easy_long_coded | stock/text | 64 | 0.734 | 0.746 | 0 | 0 | 0 |
| easy_long_coded | weather/image | 64 | 0.484 | 0.507 | 0 | 0 | 0 |
| easy_long_coded | weather/text | 64 | 0.969 | 0.944 | 0 | 0 | 0 |
| easy_short | color/image | 64 | 0.531 | 0.525 | 0 | 0 | 0 |
| easy_short | color/text | 64 | 0.531 | 0.532 | 0 | 0 | 0 |
| easy_short | medical/image | 64 | 0.734 | 0.734 | 0 | 0 | 0 |
| easy_short | medical/text | 64 | 0.859 | 0.862 | 0 | 0 | 0 |
| easy_short | news/image | 64 | 0.781 | 0.775 | 0 | 0 | 0 |
| easy_short | news/text | 64 | 0.688 | 0.672 | 0 | 0 | 0 |
| easy_short | pet/image | 64 | 0.469 | 0.471 | 0 | 0 | 0 |
| easy_short | pet/text | 64 | 0.453 | 0.458 | 0 | 0 | 0 |
| easy_short | sports/image | 64 | 0.672 | 0.667 | 0 | 0 | 0 |
| easy_short | sports/text | 64 | 0.531 | 0.534 | 0 | 0 | 0 |
| easy_short | stock/image | 64 | 0.422 | 0.423 | 0 | 0 | 0 |
| easy_short | stock/text | 64 | 0.594 | 0.580 | 0 | 0 | 0 |
| easy_short | weather/image | 64 | 0.531 | 0.524 | 0 | 0 | 0 |
| easy_short | weather/text | 64 | 0.547 | 0.543 | 0 | 0 | 0 |
| easy_short_coded | color/image | 64 | 0.656 | 0.666 | 0 | 0 | 0 |
| easy_short_coded | color/text | 64 | 1.000 | 0.999 | 0 | 0 | 0 |
| easy_short_coded | medical/image | 64 | 0.828 | 0.818 | 0 | 0 | 0 |
| easy_short_coded | medical/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | news/image | 64 | 0.734 | 0.743 | 0 | 0 | 0 |
| easy_short_coded | news/text | 64 | 1.000 | 0.988 | 0 | 0 | 0 |
| easy_short_coded | pet/image | 64 | 0.672 | 0.670 | 0 | 0 | 0 |
| easy_short_coded | pet/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | sports/image | 64 | 0.656 | 0.655 | 0 | 0 | 0 |
| easy_short_coded | sports/text | 64 | 1.000 | 0.999 | 0 | 0 | 0 |
| easy_short_coded | stock/image | 64 | 0.641 | 0.634 | 0 | 0 | 0 |
| easy_short_coded | stock/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| easy_short_coded | weather/image | 64 | 0.531 | 0.541 | 0 | 0 | 0 |
| easy_short_coded | weather/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_counts | color/image | 64 | 0.641 | 0.671 | 0 | 0 | 0 |
| explicit_counts | color/text | 64 | 1.000 | 0.998 | 0 | 0 | 0 |
| explicit_counts | medical/image | 64 | 0.875 | 0.836 | 0 | 0 | 0 |
| explicit_counts | medical/text | 64 | 0.984 | 0.967 | 0 | 0 | 0 |
| explicit_counts | news/image | 64 | 0.719 | 0.697 | 0 | 0 | 0 |
| explicit_counts | news/text | 64 | 0.859 | 0.879 | 0 | 0 | 0 |
| explicit_counts | pet/image | 64 | 0.734 | 0.727 | 0 | 0 | 0 |
| explicit_counts | pet/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_counts | sports/image | 64 | 0.641 | 0.647 | 0 | 0 | 0 |
| explicit_counts | sports/text | 64 | 1.000 | 0.934 | 0 | 0 | 0 |
| explicit_counts | stock/image | 64 | 0.438 | 0.441 | 0 | 0 | 0 |
| explicit_counts | stock/text | 64 | 1.000 | 0.934 | 0 | 0 | 0 |
| explicit_counts | weather/image | 64 | 0.516 | 0.526 | 0 | 0 | 0 |
| explicit_counts | weather/text | 64 | 1.000 | 0.976 | 0 | 0 | 0 |
| explicit_rel | color/image | 64 | 0.703 | 0.755 | 0 | 0 | 0 |
| explicit_rel | color/text | 64 | 0.984 | 0.991 | 0 | 0 | 0 |
| explicit_rel | medical/image | 64 | 0.812 | 0.818 | 0 | 0 | 0 |
| explicit_rel | medical/text | 64 | 0.906 | 0.927 | 0 | 0 | 0 |
| explicit_rel | news/image | 64 | 0.812 | 0.784 | 0 | 0 | 0 |
| explicit_rel | news/text | 64 | 0.984 | 0.945 | 0 | 0 | 0 |
| explicit_rel | pet/image | 64 | 0.453 | 0.500 | 0 | 0 | 0 |
| explicit_rel | pet/text | 64 | 1.000 | 0.989 | 0 | 0 | 0 |
| explicit_rel | sports/image | 64 | 0.719 | 0.726 | 0 | 0 | 0 |
| explicit_rel | sports/text | 64 | 1.000 | 0.995 | 0 | 0 | 0 |
| explicit_rel | stock/image | 64 | 0.562 | 0.564 | 0 | 0 | 0 |
| explicit_rel | stock/text | 64 | 1.000 | 0.992 | 0 | 0 | 0 |
| explicit_rel | weather/image | 64 | 0.484 | 0.495 | 0 | 0 | 0 |
| explicit_rel | weather/text | 64 | 1.000 | 0.934 | 0 | 0 | 0 |
| perception | color | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | medical | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | news | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | pet | 73 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | sports | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | stock | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| perception | weather | 2 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | color/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | medical/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | news/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | sports/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | stock/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| single_trial_who | weather/text | 64 | 1.000 | 1.000 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | nan | not run |
| explicit_rel | 0.8 | 0.816 | pass |
| explicit_rel by domain/format | | | color/image: fail, color/text: pass, medical/image: pass, medical/text: pass, news/image: pass, news/text: pass, pet/image: fail, pet/text: pass, sports/image: fail, sports/text: pass, stock/image: fail, stock/text: pass, weather/image: fail, weather/text: pass |

## dev_binding (2944 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| composite_read | panel1 | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| composite_read | panel2 | 64 | 1.000 | 1.000 | 0 | 0 | 0 |
| explicit_rel | color/image_after | 64 | 0.906 | 0.854 | 0 | 0 | 0 |
| explicit_rel | color/image_composite | 64 | 0.750 | 0.767 | 0 | 0 | 0 |
| explicit_rel | color/image_named | 64 | 0.625 | 0.663 | 0 | 0 | 0 |
| explicit_rel | medical/image_after | 64 | 0.562 | 0.560 | 0 | 0 | 0 |
| explicit_rel | medical/image_composite | 64 | 0.859 | 0.818 | 0 | 0 | 0 |
| explicit_rel | medical/image_named | 64 | 0.938 | 0.894 | 0 | 0 | 0 |
| explicit_rel | news/image_after | 64 | 0.516 | 0.516 | 0 | 0 | 0 |
| explicit_rel | news/image_composite | 64 | 1.000 | 0.928 | 0 | 0 | 0 |
| explicit_rel | news/image_named | 64 | 1.000 | 0.944 | 0 | 0 | 0 |
| explicit_rel | pet/image_after | 64 | 0.500 | 0.508 | 0 | 0 | 0 |
| explicit_rel | pet/image_composite | 64 | 0.969 | 0.974 | 0 | 0 | 0 |
| explicit_rel | pet/image_named | 64 | 0.875 | 0.860 | 0 | 0 | 0 |
| explicit_rel | sports/image_after | 64 | 0.484 | 0.487 | 0 | 0 | 0 |
| explicit_rel | sports/image_composite | 64 | 0.578 | 0.586 | 0 | 0 | 0 |
| explicit_rel | sports/image_named | 64 | 0.609 | 0.605 | 0 | 0 | 0 |
| explicit_rel | stock/image_after | 64 | 0.391 | 0.409 | 0 | 0 | 0 |
| explicit_rel | stock/image_composite | 64 | 0.984 | 0.984 | 0 | 0 | 0 |
| explicit_rel | stock/image_named | 64 | 0.562 | 0.564 | 0 | 0 | 0 |
| explicit_rel | weather/image_after | 64 | 0.391 | 0.401 | 0 | 0 | 0 |
| explicit_rel | weather/image_composite | 64 | 0.531 | 0.569 | 0 | 0 | 0 |
| explicit_rel | weather/image_named | 64 | 0.422 | 0.460 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | nan | not run |
| explicit_rel | 0.8 | 0.688 | fail |
| explicit_rel by domain/format | | | color/image_after: pass, color/image_composite: fail, color/image_named: fail, medical/image_after: fail, medical/image_composite: pass, medical/image_named: pass, news/image_after: fail, news/image_composite: pass, news/image_named: pass, pet/image_after: fail, pet/image_composite: pass, pet/image_named: pass, sports/image_after: fail, sports/image_composite: fail, sports/image_named: fail, stock/image_after: fail, stock/image_composite: pass, stock/image_named: fail, weather/image_after: fail, weather/image_composite: fail, weather/image_named: fail |

## dev_chartread (384 rows)

| control | stratum | items | hard acc | mean P (twin) | un-twinned | conflicts | approx |
|---|---|---|---|---|---|---|---|
| chart_read | all | 192 | 0.609 | 0.613 | 0 | 0 | 0 |

| gate | threshold | hard acc | status |
|---|---|---|---|
| chart_read | 0.8 | 0.609 | fail |
| explicit_rel | 0.8 | nan | not run |

