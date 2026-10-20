"""Orthographic particle movie from a scenario states.npz. No teleoperation."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def frame(position, codes, blade, radius, title, size=720):
    image = Image.new("RGB", (size, size), (244, 242, 236))
    draw = ImageDraw.Draw(image)
    # Look from the front-left, slightly above the board.
    view = np.array([[0.85, -0.45, 0.0], [-0.2, -0.35, 0.9]])
    projected = position @ view.T
    order = np.argsort(position @ np.array([0.2, 0.9, 0.0]))
    lo = projected.min(0)
    hi = projected.max(0)
    span = max(hi - lo)
    scale = (size * 0.78) / span
    origin = np.array([size * 0.5, size * 0.58]) - (lo + hi) * 0.5 * scale * np.array([1, -1])
    screen = np.empty_like(projected)
    screen[:, 0] = origin[0] + projected[:, 0] * scale
    screen[:, 1] = origin[1] - projected[:, 1] * scale
    radius_px = max(2, int(radius * scale * 0.9))
    palette = {
        0: (232, 224, 208),
        1: (214, 164, 92),
        2: (92, 158, 176),
    }
    for index in order:
        code = int(codes[index])
        color = palette.get(code % 3, (180, 120, 90))
        x, y = screen[index]
        draw.ellipse((x - radius_px, y - radius_px, x + radius_px, y + radius_px), fill=color)
    if blade is not None and np.isfinite(blade).all():
        tip = blade @ view.T
        bx = origin[0] + tip[0] * scale
        by = origin[1] - tip[1] * scale
        draw.line((bx, by - 28, bx, by + 36), fill=(40, 44, 52), width=4)
    draw.rectangle((0, 0, size, 36), fill=(24, 28, 34))
    draw.text((12, 10), title, fill="white")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--fps", type=float, default=25)
    args = parser.parse_args()
    with np.load(args.states) as data:
        position = data["position"]
        codes = data["sides"]
        blade = data["blade_position"][:, 0] if "blade_position" in data else None
        reference = data["reference"]
    gap = np.diff(np.unique(np.round(reference[:, 0], 6)))
    spacing = float(gap[gap > 1e-6].min()) if np.any(gap > 1e-6) else 0.005
    import imageio.v2 as imageio

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=args.fps, macro_block_size=2) as writer:
        for i in range(len(position)):
            time_s = i * 0.04
            picture = frame(
                position[i],
                codes[i],
                None if blade is None else blade[i],
                spacing / 4,
                f"{args.title} | t={time_s:.2f}s | prescribed knife",
            )
            writer.append_data(np.asarray(picture))
    print(output, len(position), flush=True)


if __name__ == "__main__":
    main()
