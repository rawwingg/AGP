
import json
import math
from pathlib import Path
import random
import numpy as np
import shapely
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union
from shapely.validation import make_valid
from generate_polygons import add_holes_to_polygon, generate_polygon
import itertools


class Coord:

    def __init__(self, coord, pairCoord):

        self.coord = coord

        self.pairCoord = pairCoord

        if coord[0]<pairCoord[0]:
            self.isLeft = True
        elif coord[0]>pairCoord[0]:
            self.isLeft = False
        else:
            self.isLeft = None


def algorithm_implementation(polygon_data, guard_radius = None, error_tolerance = 0.01, epsilon = 1):
    """
    Implement the algorithm to determine the optimal placement of guards within a polygon with holes.
    This function should return a list of guard positions that maximize coverage of the polygon area.
    """
    
    # Initialize the MWU algorithm parameters

    guards = {}

    t = (1/(epsilon**2)) * math.log(1/error_tolerance)  # number of iterations based on error tolerance and epsilon

    # Sample thousand even points within the polygon to estimate coverage

    minx, miny, maxx, maxy = polygon_data.bounds

    x = np.linspace(minx, maxx, num=200)
    y = np.linspace(miny, maxy, num=200)

    p = list(itertools.product(x, y))

    points_within = {pt: 0 for pt in p if polygon_data.contains(Point(pt))}

    no_of_points = len(points_within)

    poly_coords = list(polygon_data.exterior.coords)

    poly_edges = [LineString(poly_coords[i], poly_coords[i+1]) for i in range(0, len(poly_coords), 2)]

    all_coords = []

    for edge in poly_edges:
        all_coords.extend([edge.coords])
        

    while len(points_within) >= round(error_tolerance * no_of_points):


        #find best next guard

        #Get polygon coverage for each guard already placed -- Easy

        if guards:

            all_edges = poly_edges.copy()

            for guard in guards:

                coverage = guards[guard]

                cov_coords = coverage.exterior.coords

                cov_edges = [LineString(cov_coords[i], cov_coords[i+1]) for i in range(0, len(cov_coords), 2)]

                guard_edges = [LineString(guard, cov_coords[i]) for i in range(0, len(cov_coords))]

                all_edges.extend(cov_edges)

                all_edges.extend(guard_edges)

                #Applying the sweep line algorithm should give us all the intersections

                #Now how do we group them together?



        

        #Got a good simplified polygon coverage for each guard

        #Now to place it on top of each other and where points intersect, creating cells

        for guard in guards:
            coverage = guards[guard]
            for pt in points_within:
                if coverage.contains(Point(pt)):
                    points_within[pt] += 1

        #Inside each cell, every point has identical weight

        #Refine the arrangement of guards



        break

        #add guard

        #reduce weights of points covered by guard

        #update points_within to remove points that are now covered


    #Convert fractional solution into real guards


    return []