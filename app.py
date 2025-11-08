from math import sqrt, degrees, atan2
from PIL import Image, ImageDraw, ImageOps

def create_line(img, start_x, start_y, end_x, end_y, line_color):
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    
    x_direction = 1 if delta_x > 0 else -1 if delta_x < 0 else 0
    y_direction = 1 if delta_y > 0 else -1 if delta_y < 0 else 0
    
    abs_delta_x = abs(delta_x)
    abs_delta_y = abs(delta_y)
    
    if abs_delta_x > abs_delta_y:
        primary_step_x, primary_step_y = x_direction, 0
        error_step, primary_axis = abs_delta_y, abs_delta_x
    else:
        primary_step_x, primary_step_y = 0, y_direction
        error_step, primary_axis = abs_delta_x, abs_delta_y
    
    current_x, current_y = start_x, start_y
    error_value = 0
    img.putpixel((current_x, current_y), line_color)
    
    for step in range(primary_axis):
        error_value += 2 * error_step
        if error_value > primary_axis:
            error_value -= 2 * primary_axis
            current_x += x_direction
            current_y += y_direction
        else:
            current_x += primary_step_x
            current_y += primary_step_y
        img.putpixel((current_x, current_y), line_color)

def create_dashed_line(img, start_x, start_y, end_x, end_y, line_color):
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    
    x_direction = 1 if delta_x > 0 else -1 if delta_x < 0 else 0
    y_direction = 1 if delta_y > 0 else -1 if delta_y < 0 else 0
    
    abs_delta_x = abs(delta_x)
    abs_delta_y = abs(delta_y)
    
    if abs_delta_x > abs_delta_y:
        primary_step_x, primary_step_y = x_direction, 0
        error_step, primary_axis = abs_delta_y, abs_delta_x
    else:
        primary_step_x, primary_step_y = 0, y_direction
        error_step, primary_axis = abs_delta_x, abs_delta_y
    
    current_x, current_y = start_x, start_y
    error_value = 0

    total_steps = primary_axis
    dash_segment = total_steps / 11
    
    for step in range(total_steps + 1):
        segment_num = int(step / dash_segment)
        if segment_num % 2 == 0 and segment_num < 11:
            img.putpixel((current_x, current_y), line_color)
        error_value += 2 * error_step
        if error_value > primary_axis:
            error_value -= 2 * primary_axis
            current_x += x_direction
            current_y += y_direction
        else:
            current_x += primary_step_x
            current_y += primary_step_y

def interpolate_points(point_a, point_b, ratio):
    return (point_a[0] * (1 - ratio) + point_b[0] * ratio,
            point_a[1] * (1 - ratio) + point_b[1] * ratio)

def calculate_distance_to_line(point_x, point_y, line_start_x, line_start_y, line_end_x, line_end_y):
    line_vector_x = line_end_x - line_start_x
    line_vector_y = line_end_y - line_start_y
    point_vector_x = point_x - line_start_x
    point_vector_y = point_y - line_start_y

    line_length = sqrt(line_vector_x*line_vector_x + line_vector_y*line_vector_y)
    point_length = sqrt(point_vector_x*point_vector_x + point_vector_y*point_vector_y)
    if line_length == 0 or point_length == 0:
        return point_length

    dot_product = line_vector_x*point_vector_x + line_vector_y*point_vector_y
    cosine_angle = dot_product / (line_length * point_length)
    if cosine_angle > 1: cosine_angle = 1
    if cosine_angle < -1: cosine_angle = -1

    sine_angle = sqrt(1 - cosine_angle*cosine_angle)
    return point_length * sine_angle

def draw_circle_octants(img, center_x, center_y, radius, circle_color):
    decision_param = 3 - 2 * radius
    x_pos, y_pos = 0, radius
    while y_pos >= x_pos:
        img.putpixel((center_x + x_pos, center_y + y_pos), circle_color)
        img.putpixel((center_x + x_pos, center_y - y_pos), circle_color)
        img.putpixel((center_x - x_pos, center_y + y_pos), circle_color)
        img.putpixel((center_x - x_pos, center_y - y_pos), circle_color)
        img.putpixel((center_x + y_pos, center_y + x_pos), circle_color)
        img.putpixel((center_x + y_pos, center_y - x_pos), circle_color)
        img.putpixel((center_x - y_pos, center_y + x_pos), circle_color)
        img.putpixel((center_x - y_pos, center_y - x_pos), circle_color)
        
        if decision_param < 0:
            decision_param = decision_param + 4 * x_pos + 6
        else:
            decision_param = decision_param + 4 * x_pos - 4 * y_pos + 10
            y_pos -= 1
        x_pos += 1

def draw_partial_circle(img, center_x, center_y, radius, circle_color):
    decision_param = 3 - 2 * radius
    x_pos, y_pos = 0, radius
    while y_pos >= x_pos:
        img.putpixel((center_x + x_pos, center_y + y_pos), circle_color)
        img.putpixel((center_x - x_pos, center_y + y_pos), circle_color)
        img.putpixel((center_x + y_pos, center_y + x_pos), circle_color)
        img.putpixel((center_x - y_pos, center_y + x_pos), circle_color)
        img.putpixel((center_x - y_pos, center_y - x_pos), circle_color)

        if decision_param < 0:
            decision_param = decision_param + 4 * x_pos + 6
        else:
            decision_param = decision_param + 4 * x_pos - 4 * y_pos + 10
            y_pos -= 1
        x_pos += 1

def draw_dashed_circle_quadrants(img, center_x, center_y, radius, circle_color):
    decision_param = 2 - 2 * radius
    x_pos, y_pos = 0, radius
    while y_pos >= 0:
        for dx, dy in [(x_pos, y_pos), (x_pos, -y_pos), (-x_pos, y_pos), (-x_pos, -y_pos)]:
            angle = degrees(atan2(dy, dx))
            if angle < 0:
                angle += 360
            if int(angle // 30) % 2 == 0:
                img.putpixel((center_x + dx, center_y + dy), circle_color)
                        
        if decision_param < 0:
            error_val = 2 * decision_param + 2 * y_pos - 1
            if error_val <= 0:
                x_pos += 1
                decision_param = decision_param + 2 * x_pos + 1
                continue
            else:
                x_pos += 1
                y_pos -= 1
                decision_param = decision_param + 2 * x_pos - 2 * y_pos + 2
                continue 
        elif decision_param > 0:
            error_val = 2 * decision_param - 2 * x_pos - 1
            if error_val <= 0:
                x_pos += 1
                y_pos -= 1
                decision_param = decision_param + 2 * x_pos - 2 * y_pos + 2
                continue 
            else:
                y_pos -= 1
                decision_param = decision_param - 2 * y_pos + 1
                continue
        else:
            x_pos += 1
            y_pos -= 1
            decision_param = decision_param + 2 * x_pos - 2 * y_pos + 2

def check_point_in_polygon(test_x, test_y, polygon_vertices):
    inside_flag = False
    vertex_count = len(polygon_vertices)
    for i in range(vertex_count):
        x1, y1 = polygon_vertices[i]
        x2, y2 = polygon_vertices[(i + 1) % vertex_count]
        if ((y1 > test_y) != (y2 > test_y)) and \
           (test_x < (x2 - x1) * (test_y - y1) / (y2 - y1 + 1e-10) + x1):
            inside_flag = not inside_flag
    return inside_flag

def find_segment_intersection(seg1_start, seg1_end, seg2_start, seg2_end):
    x1, y1 = seg1_start
    x2, y2 = seg1_end
    x3, y3 = seg2_start
    x4, y4 = seg2_end

    denominator = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denominator == 0:
        return None

    t_param = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / denominator
    u_param = ((x1 - x3)*(y1 - y2) - (y1 - y3)*(x1 - x2)) / denominator

    if 0 <= t_param <= 1 and 0 <= u_param <= 1:
        intersect_x = x1 + t_param * (x2 - x1)
        intersect_y = y1 + t_param * (y2 - y1)
        return (intersect_x, intersect_y)
    return None

def draw_external_line(img, line_start_x, line_start_y, line_end_x, line_end_y, line_color, polygon_vertices):
    intersection_points = []
    for i in range(len(polygon_vertices)):
        vertex_a = polygon_vertices[i]
        vertex_b = polygon_vertices[(i + 1) % len(polygon_vertices)]
        intersection = find_segment_intersection((line_start_x, line_start_y), (line_end_x, line_end_y), vertex_a, vertex_b)
        if intersection:
            intersection_points.append(intersection)

    intersection_points.append((line_start_x, line_start_y))
    intersection_points.append((line_end_x, line_end_y))

    intersection_points.sort(key=lambda p: (p[0] - line_start_x)**2 + (p[1] - line_start_y)**2)

    for i in range(len(intersection_points) - 1):
        point_a_x, point_a_y = intersection_points[i]
        point_b_x, point_b_y = intersection_points[i + 1]
        mid_point_x = (point_a_x + point_b_x) / 2
        mid_point_y = (point_a_y + point_b_y) / 2
        if not check_point_in_polygon(mid_point_x, mid_point_y, polygon_vertices):
            create_line(img, int(point_a_x), int(point_a_y), int(point_b_x), int(point_b_y), line_color)

def draw_bezier_curve(img, control_point1, control_point2, control_point3, control_point4, tolerance=2.0, curve_color=(255,0,0), subdivision_ratio=0.5):
    distance1 = calculate_distance_to_line(control_point2[0], control_point2[1], control_point1[0], control_point1[1], control_point4[0], control_point4[1])
    distance2 = calculate_distance_to_line(control_point3[0], control_point3[1], control_point1[0], control_point1[1], control_point4[0], control_point4[1])

    if distance1 < tolerance and distance2 < tolerance:
        create_line(img, int(control_point1[0]), int(control_point1[1]), int(control_point4[0]), int(control_point4[1]), curve_color)
        return
  
    point_ab = interpolate_points(control_point1, control_point2, subdivision_ratio)
    point_bc = interpolate_points(control_point2, control_point3, subdivision_ratio)
    point_cd = interpolate_points(control_point3, control_point4, subdivision_ratio)
    point_abc = interpolate_points(point_ab, point_bc, subdivision_ratio)
    point_bcd = interpolate_points(point_bc, point_cd, subdivision_ratio)
    point_abcd = interpolate_points(point_abc, point_bcd, subdivision_ratio)

    draw_bezier_curve(img, control_point1, point_ab, point_abc, point_abcd, tolerance, curve_color, subdivision_ratio)
    draw_bezier_curve(img, point_abcd, point_bcd, point_cd, control_point4, tolerance, curve_color, subdivision_ratio)

def draw_polygon_outline(img, vertex_list, outline_color):
    for i in range(len(vertex_list)):
        current_vertex = vertex_list[i]
        next_vertex = vertex_list[(i + 1) % len(vertex_list)]
        create_line(img, 
                current_vertex[0], current_vertex[1], 
                next_vertex[0], next_vertex[1], 
                outline_color)

def extract_polygon_edges(vertex_list):
    edge_list = []
    for i in range(len(vertex_list)):
        current_vertex = vertex_list[i]
        next_vertex = vertex_list[(i + 1) % len(vertex_list)]
        if current_vertex[1] != next_vertex[1]:
            edge_list.append((current_vertex, next_vertex))
    return tuple(edge_list)

def calculate_interpolation_ratio(edge, scanline_y):
    return (scanline_y - edge[0][1]) / (edge[1][1] - edge[0][1])
  
def compute_edge_intersection(edge, scanline_y):
    ratio = calculate_interpolation_ratio(edge, scanline_y)
    intersect_x = int(ratio * edge[1][0] + (1 - ratio) * edge[0][0])
    return (intersect_x, scanline_y)

def find_all_intersections(edges_list, scanline_y):
    intersection_points = []
    for i in range(len(edges_list)):
        if scanline_y < min(edges_list[i][0][1], edges_list[i][1][1]) or scanline_y > max(edges_list[i][0][1], edges_list[i][1][1]):
            continue
        intersection = compute_edge_intersection(edges_list[i], scanline_y)
        if intersection not in intersection_points:
            intersection_points.append(intersection)
    intersection_points.sort(key=lambda point: point[0])
    return intersection_points

def apply_texture_outside_circle(img, intersection_pairs, texture_img, circle_center, circle_radius):
    texture_width, texture_height = texture_img.size
    center_x, center_y = circle_center
    radius_squared = circle_radius ** 2
    
    for i in range(0, len(intersection_pairs) - 1, 2):
        x_start, current_y = intersection_pairs[i]
        x_end, _ = intersection_pairs[i + 1]

        if 0 <= current_y < img.height:
            for x_coord in range(x_start, x_end):
                if 0 <= x_coord < img.width:
                    if (x_coord - center_x)**2 + (current_y - center_y)**2 >= radius_squared:
                        tex_x = x_coord % texture_width
                        tex_y = current_y % texture_height
                        pixel_color = texture_img.getpixel((tex_x, tex_y))
                        img.putpixel((x_coord, current_y), pixel_color)

# Основная программа
image = Image.new('RGB', (200, 200))

# Рисование треугольника
create_line(image, 60, 130, 100, 50, (255, 0, 0))
create_line(image, 60, 130, 140, 130, (255, 0, 0))
create_line(image, 100, 50, 140, 130, (255, 0, 0))

triangle_vertices = [(60, 130), (100, 50), (140, 130)]
draw_external_line(image, 0, 6, 199, 110, (255, 0, 0), triangle_vertices)

# Рисование пунктирных линий
create_dashed_line(image, 56, 133, 100, 45, (255, 0, 0))
create_dashed_line(image, 56, 133, 144, 133, (255, 0, 0))
create_dashed_line(image, 100, 45, 144, 133, (255, 0, 0))

# Рисование кругов
draw_circle_octants(image, 100, 100, 20, (255, 0, 0))
draw_dashed_circle_quadrants(image, 100, 100, 17, (255, 0, 0))
draw_partial_circle(image, 100, 100, 90, (255, 0, 0))

# Заполнение текстурой
triangle_edges = extract_polygon_edges(triangle_vertices)
circle_center_point = (100, 100)
circle_radius_value = 20

texture_image = Image.open(r"1.jpeg").convert("RGB")

min_y = min(vertex[1] for vertex in triangle_vertices)
max_y = max(vertex[1] for vertex in triangle_vertices)

for y_coord in range(min_y, max_y):
    intersections = find_all_intersections(triangle_edges, y_coord)
    if len(intersections) >= 2:
        apply_texture_outside_circle(image, intersections, texture_image, circle_center_point, circle_radius_value)

# Повторное рисование контуров
draw_circle_octants(image, 100, 100, 20, (255, 0, 0))
create_line(image, 60, 130, 100, 50, (255, 0, 0))

image.show()