# hexlab

[![PyPI version](https://img.shields.io/pypi/v/hexlab.svg)](https://pypi.org/project/hexlab/)
[![Python versions](https://img.shields.io/pypi/pyversions/hexlab.svg)](https://pypi.org/project/hexlab/)
[![Downloads](https://static.pepy.tech/badge/hexlab)](https://pepy.tech/project/hexlab)

A professional, feature-rich hex color exploration and manipulation tool for the command line.

## Introduction

**hexlab** is a powerful CLI utility for developers, designers, and accessibility experts. It provides deep insight into 24-bit colors, supporting advanced color spaces like **OKLAB, OKLCH, CIE LAB, CIE XYZ**, and standard formats like RGB, HSL, and CMYK.

Beyond inspection, hexlab offers sophisticated manipulation capabilities, including gradient generation using interpolation in perceptual color spaces, vision simulation for color blindness, and a robust adjustment pipeline for fine-tuning colors.

## Installation

hexlab requires **Python 3.7+**.

### Via PyPI (Recommended)

```bash
pip install hexlab
```

### From Source

```bash
git clone https://github.com/mallikmusaddiq1/hexlab.git
cd hexlab
pip install .
```

## Quick Start

Inspect a hex color with visual bars and WCAG contrast:

```bash
hexlab -H FF5733 -rgb -hsl -wcag
```

Generate a smooth gradient in OKLAB space:

```bash
hexlab gradient -H FF0000 -H 0000FF -cs oklab --steps 15
```

Simulate color blindness (Deuteranopia):

```bash
hexlab vision -cn "chartreuse" -d
```

## Main Command

The base command allows you to inspect a single color, view its neighbors, and retrieve technical specifications in various formats.

```bash
usage: hexlab [-h] [-H HEX | -r | -cn NAME | -di INDEX] [OPTIONS...]
```

### Input Options

Exactly one input method is required.

| Flag                  | Description |
|-----------------------|-------------|
| `-H, --hex`           | 6-digit hex color code (e.g., `FF5500`). Do not include the `#`. |
| `-r, --random`        | Generate a random 24-bit color. |
| `-cn, --color-name`   | Use a named color (e.g., `tomato`, `azure`). See `--list-color-names`. |
| `-di, --decimal-index`| Input integer index (0 to 16777215). Useful for programmatic iteration. |
| `-s, --seed`          | Seed for random generation reproducibility. |

### Navigation & Modifications

| Flag             | Description |
|------------------|-------------|
| `-n, --next`     | Show the next color (index + 1). |
| `-p, --previous` | Show the previous color (index - 1). |
| `-N, --negative` | Show the inverse (negative) color. |

### Technical Information Flags

Toggle specific color space outputs or information blocks.

| Flag            | Description |
|-----------------|-------------|
| `-all`          | Show **all** available technical information. |
| `-wcag`         | Show WCAG contrast ratios (AA/AAA) against Black and White. |
| `-hb, --hide-bars` | Hide the visual ANSI color bars (raw text output only). |
| `-rgb`          | Red, Green, Blue (0-255). |
| `-hsl`          | Hue, Saturation, Lightness. |
| `-hsv`          | Hue, Saturation, Value. |
| `-hwb`          | Hue, Whiteness, Blackness. |
| `-cmyk`         | Cyan, Magenta, Yellow, Key (Black). |
| `-lab`          | CIE 1976 Lab (Perceptually uniform). |
| `--oklab`       | OKLAB (Improved perceptual uniformity). |
| `--oklch`       | OKLCH (Cylindrical form of OKLAB). |
| `-xyz`          | CIE 1931 XYZ. |
| `-lch`          | CIE LCH (Cylindrical Lab). |
| `--cieluv`      | CIE 1976 LUV. |
| `-l, --luminance` | Relative Luminance (0.0 - 1.0). |

### Meta Options

| Flag                          | Description |
|-------------------------------|-------------|
| `--list-color-names [fmt]`    | List all supported color names. Format can be `text`, `json`, or `prettyjson`. |
| `-hf, --help-all`            | Print help for the main command AND all subcommands. |

---

## Subcommands

hexlab features a suite of specialized tools invoked via `hexlab <subcommand>`.

### 1. Gradient

Generate interpolated color steps between two or more colors. Supports interpolation in perceptual spaces like OKLAB for smoother results.

```bash
hexlab gradient -H FF0000 -H 00FF00 -S 10 -cs oklab
```

| Option              | Description |
|---------------------|-------------|
| `-H, -cn, -di, -r`  | Input colors. **Must provide at least 2 inputs** (or use `-c` with `-r`). |
| `-S, --steps`       | Total number of steps in the gradient (default: 10). |
| `-cs, --colorspace` | Interpolation space. Choices: `srgb`, `srgblinear`, `lab`, `lch`, `oklab` (default), `oklch`, `luv`. |

### 2. Mix

Mix (average) multiple colors together. Useful for finding the midpoint or blending pigments conceptually.

```bash
hexlab mix -cn red -cn blue -a 50
```

| Option           | Description |
|------------------|-------------|
| `-a, --amount`   | Mix ratio for 2 colors (0-100%). Default 50% (perfect average). |
| `-cs, --colorspace` | Mixing space. Averaging in `srgblinear` often yields more physically accurate light mixing than `srgb`. |

### 3. Scheme

Generate standard color harmonies based on color theory wheels.

```bash
hexlab scheme -H FF5733 -triadic -hm oklch
```

| Option                  | Description |
|-------------------------|-------------|
| `-hm, --harmony-model`  | The color wheel model to rotate hue on. Choices: `hsl` (classic), `lch`, `oklch` (modern). |
| `-co, --complementary`  | 180° rotation. |
| `-sco, --split-complementary` | 150° and 210° rotations. |
| `-tr, --triadic`        | 120° and 240° rotations. |
| `-an, --analogous`      | -30° and +30° rotations. |
| `-tsq, -trc`            | Tetradic Square (90° steps) and Rectangular (60°/180°). |
| `-mch`                  | Monochromatic (Lightness variations). |
| `-cs`                   | Custom degree shift (e.g., `-cs 45`). |

### 4. Vision

Simulate various forms of Color Blindness (CVD) to test accessibility.

```bash
hexlab vision -r -all
```

| Flag                  | Description |
|-----------------------|-------------|
| `-p, --protanopia`    | Red-blind simulation. |
| `-d, --deuteranopia`  | Green-blind simulation (most common). |
| `-t, --tritanopia`    | Blue-blind simulation. |
| `-a, --achromatopsia` | Total color blindness (grayscale). |

### 5. Similar

Find perceptually similar colors by searching the 24-bit space around a base color.

```bash
hexlab similar -H 336699 -dm oklab -c 5
```

| Option                | Description |
|-----------------------|-------------|
| `-dm, --distance-metric` | Algorithm to calculate "similarity". Choices: `lab` (CIEDE2000), `oklab` (Euclidean), `rgb`. |
| `-dv, --dedup-value`  | Threshold to consider colors "different". Higher values result in more distinct results. |
| `-c, --count`         | Number of similar colors to generate. |

### 6. Distinct

Generate a palette of visually distinct colors starting from a base. Uses a greedy algorithm to maximize distance.

```bash
hexlab distinct -r -c 10 -dm oklab
```

| Option           | Description |
|------------------|-------------|
| `-c, --count`    | Number of distinct colors to find. |

### 7. Convert

Utility to convert numerical color strings between formats.

```bash
hexlab convert -f hex -t rgb -v "FF0000"
hexlab convert -f oklch -t hex -v "oklch(0.6 0.15 45deg)"
```

| Option             | Description |
|--------------------|-------------|
| `-f, --from-format`| Source format (e.g., `oklch`, `rgb`, `hex`). |
| `-t, --to-format`  | Target format. |
| `-v, --value`      | The value string. Use quotes! |
| `-V, --verbose`    | Show input -> output format. |

### 8. Adjust

An advanced color manipulation pipeline. Operations are deterministic. By default, a fixed pipeline is used, but you can define custom order.

```bash
hexlab adjust -H 663399 --lighten 20 --rotate 15 --posterize 8
```

#### Tone & Vividness

| Flag                          | Description |
|-------------------------------|-------------|
| `--brightness / --brightness-srgb` | Adjust linear or sRGB brightness (-100% to 100%). |
| `--contrast`                  | Adjust contrast. |
| `--gamma`                     | Apply gamma correction. |
| `--exposure`                  | Adjust exposure in stops. |
| `--chroma-oklch`              | Scale chroma using OKLCH space. |
| `--vibrance-oklch`            | Smart saturation that boosts low-chroma colors more than high-chroma ones. |
| `--warm-oklab / --cool-oklab` | Shift color temperature. |
| `--target-rel-lum`            | Force the color to a specific relative luminance (0.0 - 1.0). |
| `--min-contrast-with`         | Ensure the result meets a contrast ratio against this hex code. |

#### Filters

| Flag            | Description |
|-----------------|-------------|
| `--grayscale`   | Convert to B&W. |
| `--sepia`       | Apply retro sepia filter. |
| `--invert`      | Invert color channels. |
| `--posterize`   | Reduce color depth to N levels. |
| `--solarize`    | Solarize effect based on OKLAB Lightness. |
| `--threshold`   | Binarize color (Black/White) based on luminance. |
| `--tint`        | Tint towards a specific Hex color. |

#### Pipeline Control

| Flag                    | Description |
|-------------------------|-------------|
| `-cp, --custom-pipeline`| Apply adjustments exactly in the order flags are passed in CLI (disables fixed pipeline). |
| `-V, --verbose`         | Log every step of the adjustment pipeline. |

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

Please ensure any new color math is backed by tests in the `tests/` directory.

## Author & License

**hexlab** is developed and maintained by:

### Mallik Mohammad Musaddiq

Email: [mallikmusaddiq1@gmail.com](mailto:mallikmusaddiq1@gmail.com)

© 2025 Hexlab. Open Source Software.

