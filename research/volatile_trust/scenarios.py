"""Cover-story scenarios. Every scenario is the same abstract task: two named sources give conflicting
binary reports; the truth is revealed afterwards. Only wording, answer words and icons change.

The 'color' and 'pet' templates reproduce the original (v1/v2) prompts word for word.
All answer words are single tokens in the Qwen2.5-VL / Qwen3-VL / InternVL3.5 tokenizers (checked 2026-11-12).
"""
from PIL import Image, ImageDraw

SCENARIOS = {
    'color': dict(labels=('red', 'blue'), unit='trial', roles='reporters', role='reporter', verb='reports',
                  setup='give conflicting reports about a hidden color',
                  reveal='the true color is revealed', outcome='True color',
                  pending='true color not yet revealed',
                  question='Predict the true color on the current trial.', noun='color'),
    'pet': dict(labels=('cat', 'dog'), unit='trial', roles='reporters', role='reporter', verb='reports',
                setup='give conflicting reports about a hidden animal',
                reveal='the true animal is revealed', outcome='True animal',
                pending='true animal not yet revealed',
                question='Predict the true animal on the current trial.', noun='animal'),
    'weather': dict(labels=('rain', 'dry'), unit='day', roles='weather forecasters', role='forecaster',
                    verb='forecasts', setup="give conflicting forecasts for the day's weather (rain or dry)",
                    reveal='the actual weather is observed', outcome='Actual weather',
                    pending='actual weather not yet observed',
                    question='Predict the actual weather on the current day.', noun='weather'),
    'stock': dict(labels=('up', 'down'), unit='week', roles='stock analysts', role='analyst', verb='predicts',
                  setup='give conflicting predictions about whether a stock will go up or down that week',
                  reveal='the actual move of the stock is observed', outcome='Actual move',
                  pending='actual move not yet observed',
                  question='Predict the actual move of the stock in the current week.', noun='move'),
    'sports': dict(labels=('win', 'lose'), unit='game', roles='sports commentators', role='commentator',
                   verb='predicts', setup='give conflicting predictions about whether the home team will win or lose',
                   reveal='the actual result is observed', outcome='Actual result',
                   pending='actual result not yet observed',
                   question='Predict the actual result of the current game.', noun='result'),
    'medical': dict(labels=('positive', 'negative'), unit='case', roles='doctors', role='doctor', verb='reads',
                    setup="read the same patient's scan and give conflicting readings (positive or negative)",
                    reveal='a definitive test reveals the true result', outcome='True result',
                    pending='true result not yet known',
                    question='Predict the true result of the current case.', noun='result'),
    'news': dict(labels=('true', 'false'), unit='story', roles='news sources', role='source', verb='says',
                 setup='make conflicting claims about whether a rumor is true or false',
                 reveal='the rumor is verified', outcome='Verified',
                 pending='not yet verified', question='Predict whether the current rumor is true or false.',
                 noun='verdict'),
}
TEXT_SCENARIOS = ['color', 'weather', 'stock', 'sports', 'medical', 'news']   # rotated across episodes
COLOR_RGB = {'red': (220, 30, 30), 'blue': (30, 60, 220)}


def plural(unit):
    return {'story': 'stories'}.get(unit, unit + 's')


ICON_VERSION = 'v3.1'
# v3.1 (2026-11-12, deviation D1 in EXPERIMENTS.md): the v3 medical test-strip and sports trophy/cross icons were
# illegible to the models (single-image perception at .50 on development items). They are replaced by a plus/minus
# sign (medical) and a happy/sad face (sports), chosen on development control items only (icon_probe.py).
# All other icons are unchanged. icon(..., version='v3') reproduces the original icons.


def icon(scenario, lab, P, version=ICON_VERSION):
    """Deterministic, text-free icon for a scenario label (P x P pixels)."""
    if version == 'v3.1' and scenario in ('medical', 'sports'):
        return _icon_v31(scenario, lab, P)
    img = Image.new('RGB', (P, P), 'white')
    d = ImageDraw.Draw(img)
    s = P / 224

    def S(*xy):
        return [v * s for v in xy]
    if scenario == 'color':
        return Image.new('RGB', (P, P), COLOR_RGB[lab])
    if scenario == 'weather':
        if lab == 'dry':                                   # sun
            d.ellipse(S(72, 72, 152, 152), fill=(250, 200, 30))
            for a, b, c, e in ((112, 20, 112, 56), (112, 168, 112, 204), (20, 112, 56, 112), (168, 112, 204, 112),
                               (45, 45, 70, 70), (154, 154, 179, 179), (45, 179, 70, 154), (154, 70, 179, 45)):
                d.line(S(a, b, c, e), fill=(250, 200, 30), width=int(8 * s))
        else:                                              # cloud with rain drops
            for box in ((40, 60, 120, 130), (90, 40, 180, 130), (60, 80, 190, 140)):
                d.ellipse(S(*box), fill=(120, 130, 145))
            for x in (70, 105, 140, 175):
                d.line(S(x, 155, x - 12, 195), fill=(40, 90, 220), width=int(7 * s))
    elif scenario == 'stock':
        if lab == 'up':
            d.line(S(30, 180, 90, 130, 130, 150, 190, 50), fill=(30, 160, 60), width=int(10 * s))
            d.polygon(S(190, 40, 160, 55, 192, 80), fill=(30, 160, 60))
        else:
            d.line(S(30, 50, 90, 100, 130, 80, 190, 180), fill=(210, 40, 40), width=int(10 * s))
            d.polygon(S(192, 190, 160, 175, 190, 150), fill=(210, 40, 40))
    elif scenario == 'sports':
        if lab == 'win':                                   # trophy
            d.polygon(S(62, 40, 162, 40, 150, 110, 112, 130, 74, 110), fill=(220, 175, 30))
            d.rectangle(S(102, 128, 122, 165), fill=(220, 175, 30))
            d.rectangle(S(75, 165, 149, 185), fill=(150, 110, 20))
        else:                                              # broken/downturned thumb-free: red cross on gray ball
            d.ellipse(S(52, 52, 172, 172), fill=(170, 170, 170))
            d.line(S(70, 70, 154, 154), fill=(210, 40, 40), width=int(14 * s))
            d.line(S(154, 70, 70, 154), fill=(210, 40, 40), width=int(14 * s))
    elif scenario == 'medical':                            # lateral-flow test strip: 2 lines = positive
        d.rounded_rectangle(S(80, 20, 144, 204), radius=int(10 * s), fill=(235, 235, 235), outline=(90, 90, 90),
                            width=int(3 * s))
        d.rectangle(S(92, 60, 132, 150), fill=(255, 255, 255), outline=(150, 150, 150))
        d.line(S(92, 85, 132, 85), fill=(200, 30, 60), width=int(6 * s))          # control line
        if lab == 'positive':
            d.line(S(92, 120, 132, 120), fill=(200, 30, 60), width=int(6 * s))    # test line
    elif scenario == 'news':
        if lab == 'true':
            d.line(S(45, 115, 95, 165, 180, 60), fill=(30, 160, 60), width=int(18 * s))
        else:
            d.line(S(55, 55, 170, 170), fill=(210, 40, 40), width=int(18 * s))
            d.line(S(170, 55, 55, 170), fill=(210, 40, 40), width=int(18 * s))
    else:
        raise ValueError(scenario)
    return img


def _icon_v31(scenario, lab, P):
    img = Image.new('RGB', (P, P), 'white')
    d = ImageDraw.Draw(img)
    s = P / 224

    def S(*xy):
        return [v * s for v in xy]
    if scenario == 'medical':                               # thick plus (positive) / minus (negative) in a ring
        d.ellipse(S(22, 22, 202, 202), outline=(80, 80, 80), width=int(6 * s))
        d.rectangle(S(52, 98, 172, 126), fill=(60, 60, 60))
        if lab == 'positive':
            d.rectangle(S(98, 52, 126, 172), fill=(60, 60, 60))
    else:                                                   # happy face (win) / sad face (lose)
        d.ellipse(S(32, 32, 192, 192), fill=(250, 210, 60) if lab == 'win' else (160, 170, 190))
        d.ellipse(S(75, 80, 95, 100), fill=(40, 40, 40))
        d.ellipse(S(129, 80, 149, 100), fill=(40, 40, 40))
        if lab == 'win':
            d.arc(S(70, 95, 154, 160), start=20, end=160, fill=(40, 40, 40), width=int(8 * s))
        else:
            d.arc(S(70, 125, 154, 185), start=200, end=340, fill=(40, 40, 40), width=int(8 * s))
    return img
