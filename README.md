# hexlab

[![PyPI version](https://img.shields.io/pypi/v/hexlab.svg)](https://pypi.org/project/hexlab/)
[![Python versions](https://img.shields.io/pypi/pyversions/hexlab.svg)](https://pypi.org/project/hexlab/)
[![Downloads](https://img.shields.io/pypi/dm/hexlab.svg)](https://pypi.org/project/hexlab/)

**hexlab** is a 24‑bit hex color exploration and manipulation tool for the terminal.  
It lets you inspect a color in multiple color spaces, see neighbors and negatives, and check WCAG contrast, all with truecolor previews and visual component bars.

---

## Features

- 24‑bit color space support from `#000000` to `#FFFFFF` (0 to `MAX_DEC`).
- Multiple ways to pick a color:
  - Direct hex input (`-H/--hex`).
  - Random color generation (`-r/--random`) with optional seed for reproducibility (`-s/--seed`).
  - Named colors (`-cn/--color-name`) backed by a color name database.
  - Decimal index input (`-di/--decimal-index`) over the full 24‑bit range.
- Rich technical output:
  - Relative luminance and WCAG contrast against white and black.
  - RGB, HSL, HSV, HWB, CMYK.
  - CIE XYZ, CIE Lab, CIE LCh, CIE Luv.
  - OKLab and OKLCh.
- Visual terminal output:
  - Truecolor swatches rendered directly in the terminal.
  - Colored bar graphs for each component (R/G/B, H/S/L, etc.), with automatic clamping and near‑zero cleanup.
- Navigation in color space:
  - `--next`, `--previous` to step through indices.
  - `--negative` to show the inverse color.
- Color name tooling:
  - `--list-color-names` in `text`, `json`, or `prettyjson` formats.
  - `--name` to show the resolved color name for the current color, if known.
- Extensible subcommand system:
  - `hexlab <subcommand> ...` dispatches to additional tools shipped as submodules.
  - `--help-full` prints help for the main CLI and all registered subcommands.

---

## Installation

Install from PyPI:

```
pip install hexlab
```

You can also install from source:

```
git clone https://github.com/mallikmusaddiq1/hexlab.git
cd hexlab
pip install .
```

---

## Quick start

Show basic info for a specific hex color:

```
hexlab -H FF5733 -rgb -hsl -wcag --name
```

Generate a random color and show all technical information:

```
hexlab -r --all-tech-infos
```

Look up a named color and inspect it in multiple color spaces:

```
hexlab -cn "dodgerblue" -rgb -hsl -oklab -oklch
```

Work with decimal indices instead of hex codes:

```
hexlab -di 16711680 -rgb -luminance
```

Explore neighboring and inverse colors:

```
hexlab -H 00FF00 -n -p -N -rgb -hsl
```

List all available color names (for use with `-cn/--color-name`):

```
hexlab --list-color-names           # text
hexlab --list-color-names json      # machine-readable
hexlab --list-color-names prettyjson
```

---

## CLI usage

### Basic syntax

```
hexlab [OPTIONS]
hexlab <subcommand> [OPTIONS]
```

If the first argument matches a known subcommand, `hexlab` dispatches directly to that subcommand.  
Otherwise, it treats the arguments as options for the main color inspection command.

### Color selection options

Exactly one of these is required for the main color command:

- `-H HEX`, `--hex HEX`  
  6‑digit hex color code without the `#` sign (e.g. `FFAA00`).

- `-r`, `--random`  
  Generate a random 24‑bit hex color.

- `-cn NAME`, `--color-name NAME`  
  Color name from the database (`hexlab --list-color-names` to see valid names).

- `-di INDEX`, `--decimal-index INDEX`  
  Decimal index of the color, from `0` to `MAX_DEC` (full 24‑bit range).

Random generation can be made reproducible with:

- `-s SEED`, `--seed SEED`  
  Seed for the internal RNG so repeated runs give the same random color.

### Color navigation options

These operate on the current base color (no extra arguments):

- `-n`, `--next`  
  Show the next color (`index + 1` modulo the 24‑bit range).

- `-p`, `--previous`  
  Show the previous color (`index - 1` modulo the 24‑bit range).

- `-N`, `--negative`  
  Show the inverse color (`MAX_DEC - index`).

When enabled, neighbors are shown as additional truecolor swatches, each with its own label (e.g. `next`, `previous`, `negative`).

---

## Technical output flags

### Meta and general controls

- `-i`, `--index`  
  Show the decimal index of the current color.

- `--name`  
  Print the resolved color name if the color exists in the name database.

- `-all`, `--all-tech-infos`  
  Enable all technical information flags at once by setting all keys from the internal `TECH_INFO_KEYS`.

- `--hide-bars`  
  Suppress the colored bar graphs and show only raw values.

### Color space and metric flags

Each flag adds one block of information to the output:

- `-rgb`, `--red-green-blue`  
  RGB triplet and per‑channel bars.

- `-l`, `--luminance`  
  Relative luminance of the color.

- `-hsl`, `--hue-saturation-lightness`  
  HSL coordinates plus bar graphs for H, S, and L.

- `-hsv`, `--hue-saturation-value`  
  HSV coordinates with component bars.

- `-hwb`, `--hue-whiteness-blackness`  
  HWB coordinates with bars for whiteness and blackness.

- `-cmyk`, `--cyan-magenta-yellow-key`  
  CMYK values and per‑channel bars.

- `-xyz`, `--ciexyz`  
  CIE 1931 XYZ values; bars are normalized to `[0, 1]`.

- `-lab`, `--cielab`  
  CIE 1976 Lab coordinates.  
  Small numerical noise in a and b components is cleaned up near zero for nicer output.

- `-lch`, `--lightness-chroma-hue`  
  LCh representation derived from Lab.

- `--cieluv`, `-luv`  
  CIE 1976 Luv coordinates, also with near‑zero cleanup for U and V.

- `--oklab`  
  OKLab coordinates with bars scaled around the typical a/b range.

- `--oklch`  
  OKLCh coordinates derived from OKLab.

- `-wcag`, `--contrast`  
  WCAG contrast ratio against both pure white and pure black, including AA/AAA status labels.  
  The output includes a truecolor background block showing how white and black text look on the current color.

---

## Help and subcommands

Standard help flags:

- `-h`, `--help`  
  Show basic help for the main CLI.

- `-v`, `--version`  
  Show the installed `hexlab` version.

Extended help including subcommands:

- `-hf`, `--help-full`  
  Print the main help, then iterate over all registered subcommands and print each subcommand’s help if available.

Subcommands are invoked by placing the command name first:

```
hexlab <subcommand> [OPTIONS...]
```

- If `<subcommand>` is recognized, `hexlab`:
  - Ensures truecolor support.
  - Dispatches to `SUBCOMMANDS[<name>].main()` in the corresponding module.
- If you pass a subcommand name later in the argument list, `hexlab` treats it as an error and prints a helpful message that the command must be the first argument.

Each subcommand module can optionally expose a `get_<name>_parser()` function to integrate with `--help-full` so that its individual parser help is shown automatically.

> You can document each subcommand in more detail in this README once those modules are finalized.

---

## Internal design notes

For contributors and advanced users, the main workflow is:

1. **Argument parsing**  
   - Uses a custom `HexlabArgumentParser` wrapper around `argparse`, with an explicit `add_help=False` so that `-h/--help` is controlled manually.  
   - A mutually exclusive group enforces “exactly one” of hex, random, color name, or decimal index.

2. **Color selection and neighbors**  
   - Once the base color hex code is decided, its integer index is computed and neighbor indices are derived by modular arithmetic over `[0, MAX_DEC]`.  
   - Optional “next”, “previous”, and “negative” colors are collected into a `neighbors` dict and passed to the renderer.

3. **Color math pipeline**  
   - All conversions start from the RGB triplet.  
   - Intermediate XYZ and Lab values are only computed if needed (e.g. if Lab or LCh output is requested) to avoid unnecessary work.  
   - OKLab and OKLCh are derived via dedicated conversion functions, keeping implementation isolated from the CLI layer.

4. **Output rendering**  
   - `print_color_block` prints large truecolor swatches for the base color and any neighbors.  
   - `_draw_bar` builds 16‑character colored bars with filled (`█`) and empty (`░`) segments, using ANSI 24‑bit foreground colors and a dim style for the empty portion.  
   - `_zero_small` clamps tiny floating‑point noise to zero so that Lab, Luv, OKLab, and OKLCh outputs look clean.

5. **Contrast and WCAG checks**  
   - Relative luminance is computed in a dedicated module, then passed to a WCAG contrast helper that returns detailed ratios and pass/fail levels for AA/AAA.  
   - The CLI prints a compact three‑line block: white text on the color, a separator row, and black text on the color, each annotated with its contrast ratio and conformance levels.

This structure keeps the CLI logic thin, with the heavy lifting done by specialized modules in `color_math`, `utils`, and `constants`.

---

## Contributing

Contributions are welcome, whether they are bug fixes, new color spaces, better formatting, or new subcommands.

Typical workflow:

1. Fork the repository and create a feature branch:

   ```
   git checkout -b feature/my-improvement
   ```

2. Make changes with the following guidelines:
   - Keep CLI behavior consistent and user‑friendly (clear error messages via the logging helper).
   - Add or update unit tests if you add new conversions or behavior.
   - Prefer using existing helpers in `color_math`, `utils`, and `constants` rather than duplicating logic.

3. If you add a new subcommand:
   - Implement a module under the `subcommands` package exposing at least a `main()` function.
   - Register it in the central `SUBCOMMANDS` mapping.
   - Optionally add a `get_<name>_parser()` function so that `--help-full` can display its dedicated help.

4. Run tests and basic sanity checks (e.g. run `hexlab --help`, `hexlab --help-full`, and a few sample commands) before opening a pull request.

5. Open a PR with a clear description of the change and relevant usage examples.

---

## Author

Name: **Mallik Mohammad Musaddiq**  

Email: [mallikmusaddiq1@gmail.com](mailto:mallikmusaddiq1@gmail.com)

[**GitHub Profile**](github.com/mallikmusaddiq1)

---

## License

This project is intended to include a standard open‑source license.  
Add a `LICENSE` file to the repository and update this section to match the chosen license.