import svgpathtools
from svgpathtools import svg2paths2
import os

# Function to convert an SVG path command to G-code
def svg_path_to_gcode(path, scale=1.0, feed_rate=10000):
    gcode = []
    for segment in path:
        if isinstance(segment, svgpathtools.Line):
            start = segment.start
            end = segment.end
            gcode.append(f"G1 X{end.real * scale:.2f} Y{end.imag * scale:.2f} F{feed_rate}")
        elif isinstance(segment, svgpathtools.QuadraticBezier):
            # Approximate quadratic Bezier with linear segments
            num_points = 10
            points = [segment.point(t / num_points) for t in range(num_points + 1)]
            for point in points:
                gcode.append(f"G1 X{point.real * scale:.2f} Y{point.imag * scale:.2f} F{feed_rate}")
        elif isinstance(segment, svgpathtools.CubicBezier):
            # Approximate cubic Bezier with linear segments
            num_points = 10
            points = [segment.point(t / num_points) for t in range(num_points + 1)]
            for point in points:
                gcode.append(f"G1 X{point.real * scale:.2f} Y{point.imag * scale:.2f} F{feed_rate}")
        elif isinstance(segment, svgpathtools.Arc):
            # Approximate arc with linear segments
            num_points = 10
            points = [segment.point(t / num_points) for t in range(num_points + 1)]
            for point in points:
                gcode.append(f"G1 X{point.real * scale:.2f} Y{point.imag * scale:.2f} F{feed_rate}")
    return gcode

# Function to convert an entire SVG file to G-code
def svg_to_gcode(svg_file, scale=1.0, feed_rate=10000):
    paths, attributes, svg_attributes = svg2paths2(svg_file)
    gcode = []

    for path in paths:
        gcode.append("M3 S50")  # Lift the pen
        start_point = path[0].start
        gcode.append(f"G0 X{start_point.real * scale:.2f} Y{start_point.imag * scale:.2f}")  # Move to the start of the path
        gcode.append("M3 S0")  # Lower the pen

        gcode.extend(svg_path_to_gcode(path, scale, feed_rate))

    gcode.append("G0 Z5")  # Lift the pen at the end
    return gcode

# Save G-code to a file
def save_gcode(gcode, filename):
    with open(filename, 'w') as f:
        for line in gcode:
            f.write(line + '\n')

# Directory containing SVG files
svg_directory = 'c:/Users/kilia/Desktop/COOP/Thesis/pre_svg'

# Directory to save G-code files
gcode_directory = r'c:\\Users\\kilia\\Desktop\\COOP\\Thesis\\pre_gcode'

# Create the G-code directory if it doesn't exist
os.makedirs(gcode_directory, exist_ok=True)

# Process each SVG file in the directory
for filename in os.listdir(svg_directory):
    if filename.endswith('.svg'):
        svg_file = os.path.join(svg_directory, filename)
        gcode = svg_to_gcode(svg_file, scale=0.2, feed_rate=10000)
        gcode_filename = os.path.splitext(filename)[0] + '.gcode'
        gcode_filepath = os.path.join(gcode_directory, gcode_filename)
        save_gcode(gcode, gcode_filepath)
        print(f"Converted {filename} to {gcode_filename}")