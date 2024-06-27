# Node Quick Maths


## Overview
This extension creates a dialogue that makes it simple to create long chains of math nodes in shaders (and soon geometry nodes).


## Usage

### Default Keyboard Shortcut - Ctrl + M

In the text box that appears, type in a mathematical expression that you want to evaluate. For example, 1+2*3. 
Accept the dialogue and the tree will be created with math nodes.

Mathematical operators you would expect to work are supported, including:
- Addition, Subtraction, Multiplication, Division, Power: + - * / **
- Less Than, Greater Than: < >
- Modulo: %

And all other mathematical functions in the Math node is supported.
- Logarithm: log(x[, base])
- Square Root: sqrt(x)
- Abs: abs(x)
- Exponential: exp(x)
### Comparison
- Min: min(x, y)
- Max: max(x, y)
- Compare: cmp(x, y, z)
- Smooth Min: smin(x, y, z)
- Smooth Max: smax(x, y, z)
### Rounding
- Round: round(x)
- Floor: floor(x)
- Ceil: ceil(x)
- Truncate: trunc(x) or int(x)
- Fraction: frac(x)
- Floored Modulo: fmod(x, y)
- Wrap: wrap(x, y, z)
- Snap: snap(x, y)
- Ping-Pong: pingpong(x, y)
### Trig
- Sin: sin(x)
- Cos: cos(x)
- Tan: tan(x)
- Arcsine: asin(x)
- Arccosine: acos(x)
- Arctangent: atan(x)
- Arctan2: atan2(x)
- Hyperbolic Sine: sinh(x)
- Hyperbolic Cosine: cosh(x)
- Hyperbolic Tangent: tanh(x)
### Conversion
- To Radians: rad(degx)
- To Degrees: deg(radx)



# Installation:

To install the node-quick-maths extension, you must have Blender 4.2+ and have enabled Extensions.
Go to https://extensions.blender.org/ and search up this project

## License

This addon is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html). See the `LICENSE` file for more details.
