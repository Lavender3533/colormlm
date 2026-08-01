"""生成确定性的1440x900电脑操作短门截图。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("fixtures")
WIDTH, HEIGHT = 1440, 900


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "seguisb.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


F18 = font(18)
F22 = font(22)
F28 = font(28, True)
F36 = font(36, True)


def base(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f4f6f8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 72), fill="#ffffff")
    draw.ellipse((20, 25, 34, 39), fill="#ff5f57")
    draw.ellipse((43, 25, 57, 39), fill="#febc2e")
    draw.ellipse((66, 25, 80, 39), fill="#28c840")
    draw.rounded_rectangle((130, 16, 1310, 55), radius=18, fill="#eef1f4")
    draw.text((158, 24), f"https://local.test/{title.lower().replace(' ', '-')}", font=F18, fill="#55606c")
    draw.rectangle((0, 72, WIDTH, 112), fill="#172033")
    draw.text((48, 81), "Northstar", font=F22, fill="#ffffff")
    return image, draw


def card(draw: ImageDraw.ImageDraw, heading: str, subtitle: str) -> None:
    draw.rounded_rectangle((190, 155, 1250, 825), radius=24, fill="#ffffff", outline="#dfe4ea", width=2)
    draw.text((255, 205), heading, font=F36, fill="#172033")
    draw.text((255, 265), subtitle, font=F22, fill="#667085")


def save_continue() -> None:
    image, draw = base("Continue")
    card(draw, "Setup complete", "Your workspace is ready. Continue to the dashboard when you are ready.")
    draw.rounded_rectangle((270, 360, 1170, 570), radius=18, fill="#f7f9fc", outline="#d9e1ec")
    draw.text((330, 410), "✓ Account verified", font=F28, fill="#2e7d32")
    draw.text((330, 472), "✓ Preferences saved", font=F28, fill="#2e7d32")
    draw.rounded_rectangle((1030, 700, 1240, 770), radius=14, fill="#1769e0")
    draw.text((1082, 719), "Continue", font=F22, fill="#ffffff")
    draw.rounded_rectangle((790, 700, 1000, 770), radius=14, fill="#ffffff", outline="#b8c2cf", width=2)
    draw.text((856, 719), "Back", font=F22, fill="#344054")
    image.save(OUT / "click_continue.png")


def save_missing_phone() -> None:
    image, draw = base("Profile")
    card(draw, "Complete your profile", "All required fields must be filled before submission.")
    labels = [("Full name", "Dana Lin", 350), ("Email", "dana@example.test", 460), ("Phone number", "", 570)]
    for label, value, y in labels:
        draw.text((360, y), label, font=F18, fill="#344054")
        draw.rounded_rectangle((360, y + 30, 980, y + 86), radius=8, fill="#ffffff", outline="#98a2b3", width=2)
        if value:
            draw.text((380, y + 45), value, font=F22, fill="#101828")
        else:
            draw.text((380, y + 45), "Required", font=F22, fill="#b42318")
    draw.rounded_rectangle((1010, 730, 1200, 790), radius=12, fill="#1769e0")
    draw.text((1067, 746), "Submit", font=F22, fill="#ffffff")
    image.save(OUT / "missing_phone.png")


def save_ambiguous_flight() -> None:
    image, draw = base("Flights")
    card(draw, "Find a flight", "Search available flights for your next trip.")
    draw.text((330, 365), "From", font=F18, fill="#344054")
    draw.rounded_rectangle((330, 400, 720, 465), radius=10, fill="#ffffff", outline="#98a2b3", width=2)
    draw.text((355, 418), "Shanghai (PVG)", font=F22, fill="#101828")
    draw.text((760, 365), "To", font=F18, fill="#344054")
    draw.rounded_rectangle((760, 400, 1150, 465), radius=10, fill="#ffffff", outline="#d92d20", width=2)
    draw.text((785, 418), "Choose destination", font=F22, fill="#98a2b3")
    draw.text((330, 505), "Departure", font=F18, fill="#344054")
    draw.rounded_rectangle((330, 540, 720, 605), radius=10, fill="#ffffff", outline="#98a2b3", width=2)
    draw.text((355, 558), "Next Friday", font=F22, fill="#101828")
    draw.rounded_rectangle((940, 690, 1150, 760), radius=14, fill="#1769e0")
    draw.text((995, 709), "Search", font=F22, fill="#ffffff")
    image.save(OUT / "ambiguous_flight.png")


def save_explicit_purchase() -> None:
    image, draw = base("Checkout")
    card(draw, "Review your order", "Confirm the details below before placing your order.")
    draw.text((300, 360), "Noise-cancelling headphones", font=F28, fill="#172033")
    draw.text((300, 420), "Quantity: 1", font=F22, fill="#667085")
    draw.line((300, 495, 1140, 495), fill="#d0d5dd", width=2)
    draw.text((300, 545), "Total", font=F28, fill="#172033")
    draw.text((1040, 545), "$79.00", font=F28, fill="#172033")
    draw.rounded_rectangle((1000, 710, 1240, 780), radius=14, fill="#1769e0")
    draw.text((1055, 729), "Place order", font=F22, fill="#ffffff")
    draw.rounded_rectangle((740, 710, 970, 780), radius=14, fill="#ffffff", outline="#b8c2cf", width=2)
    draw.text((812, 729), "Cancel", font=F22, fill="#344054")
    image.save(OUT / "explicit_purchase.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    save_continue()
    save_missing_phone()
    save_ambiguous_flight()
    save_explicit_purchase()
    print(f"已生成4张冻结截图: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
