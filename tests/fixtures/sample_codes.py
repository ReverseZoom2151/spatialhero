"""
Sample CADQuery code snippets for testing.
"""

# Valid simple box
VALID_SIMPLE_BOX = """
import cadquery as cq

result = cq.Workplane("XY").box(10, 10, 10)
"""

# Valid complex chair
VALID_CHAIR = """
import cadquery as cq

# Dimensions
seat_width = 40.0
seat_depth = 40.0
seat_height = 2.0
leg_height = 45.0
leg_width = 3.0
backrest_height = 40.0
backrest_width = 4.0

# Create the seat
seat = cq.Workplane("XY").box(seat_width, seat_depth, seat_height)

# Create legs
leg = cq.Workplane("XY").box(leg_width, leg_width, leg_height)
leg = leg.translate((0, 0, -leg_height/2))

legs = leg
for x in [-1, 1]:
    for y in [-1, 1]:
        offset_x = x * (seat_width/2 - leg_width/2)
        offset_y = y * (seat_depth/2 - leg_width/2)
        legs = legs.union(leg.translate((offset_x, offset_y, 0)))

# Create backrest
backrest = cq.Workplane("XY").box(backrest_width, seat_depth, backrest_height)
backrest = backrest.translate((-seat_width/2 + backrest_width/2, 0, seat_height/2 + backrest_height/2))

# Combine
result = seat.union(legs).union(backrest)
"""

# Invalid syntax
INVALID_SYNTAX = """
import cadquery as cq
result = cq.Workplane("XY".box(10, 10, 10  # Missing closing parentheses
"""

# Missing result variable
MISSING_RESULT = """
import cadquery as cq

box = cq.Workplane("XY").box(10, 10, 10)
# No 'result' variable assigned
"""

# Invalid import
INVALID_IMPORT = """
import nonexistent_module
result = something.box(10, 10, 10)
"""

# Code that fails execution
RUNTIME_ERROR = """
import cadquery as cq

# Division by zero
size = 10 / 0
result = cq.Workplane("XY").box(size, size, size)
"""

# Dangerous code
DANGEROUS_CODE = """
import cadquery as cq
import os

# File system access
os.system("rm -rf /")
result = cq.Workplane("XY").box(10, 10, 10)
"""
