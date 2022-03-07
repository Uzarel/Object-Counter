import numpy as np


class Point:
    """
    Defines a point within the 2D space whose origin is in the upper-left corner
    """
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'({self.x}, {self.y})'
    
    def to_tuple(self):
        return (self.x, self.y)
    

def ccw(A: Point, B: Point, C: Point):
    """
    Checks if three points are in a counter-clockwise order
    """
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


class Segment:
    """
    Defines a line segment with two given endpoints
    """
    def __init__(self, first_endpoint: Point, second_endpoint: Point):
        self.first_endpoint = first_endpoint
        self.second_endpoint = second_endpoint
    
    def __repr__(self):
        return f'[({self.first_endpoint.x}, {self.first_endpoint.y}), ({self.second_endpoint.x}, {self.second_endpoint.y})]'

    def intersect(self, segment: 'Segment'):
        """
        Checks if it intersects another segment
        """
        # assert isinstance(segment, Segment)
        A = self.first_endpoint
        B = self.second_endpoint
        C = segment.first_endpoint
        D = segment.second_endpoint
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


# TODO: Already exists within norfir.drawing
def centroid(tracked_points):
    """
    Generates a centroid from any list of points [[x1, y1], ..., [xn, yn]]
    """
    num_points = tracked_points.shape[0]
    sum_x = np.sum(tracked_points[:, 0])
    sum_y = np.sum(tracked_points[:, 1])
    return Point(int(sum_x / num_points), int(sum_y / num_points))