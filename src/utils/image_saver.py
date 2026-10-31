from io import BytesIO
from pathlib import Path

from PIL import Image


def save_optimized_figure(fig, output_path, **savefig_kwargs):
    """
    Save Matplotlib figures as lossless WebP.
    Preserve the original figure resolution and appearance.
    """
    output_path = Path(output_path).with_suffix(".webp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    savefig_kwargs.pop("format", None)

    with BytesIO() as buffer:
        fig.savefig(buffer, format="png", **savefig_kwargs)
        buffer.seek(0)

        with Image.open(buffer) as image:
            image.load()

            if image.mode not in ("RGB", "RGBA"):
                image = image.convert(
                    "RGBA" if "A" in image.getbands() else "RGB"
                )

            image.save(
                output_path,
                format="WEBP",
                lossless=True,
                method=6,
                exact=True,
            )

    return output_path
