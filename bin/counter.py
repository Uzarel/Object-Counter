from .geometry import centroid, Segment
import norfair
from typing import List, Optional


class Counter:
    """
    Defines a class that counts how many unique items, represented by their centroid, cross a given line segment
    """
    def __init__(self, line_segment: Segment):
        self.count = 0 # Number of items that crossed the line segment
        self.exhausted_ids = list() # Object IDs that already crossed the line segment
        self.object_segments = dict() # Objects segments that are tracked for line crossing evaluation
        self.line_segment = line_segment # Line segment to compute intersection with

    def update(self, tracked_objects: Optional[List['norfair.tracker.TrackedObject']] = None):
        object_ids = list() # Keeps track of tracked object ids for later object segments and exhausted ids clean-up
        # Segments update section
        if tracked_objects: # Update only if there are tracked objects available
            for tracked_object in tracked_objects:
                object_id = tracked_object.id
                object_ids.append(object_id)
                if any(tracked_object.live_points) and object_id not in self.exhausted_ids: # Update only objects that have been recently matched with detections and not exhausted already
                    estimate = centroid(tracked_object.estimate)
                    if object_id not in self.object_segments.keys(): # If tracked object is new, generate a fake segment for it
                        self.object_segments[object_id] = Segment(estimate, estimate)
                    else: # If tracked object is not new, update its segment accordingly
                        old_segment = self.object_segments[object_id]
                        self.object_segments[object_id] = Segment(old_segment.second_endpoint, estimate)
        else: # If no tracked objects are available, re-initialise object segments and exhausted ids
            self.exhausted_ids = list()
            self.object_segments = dict()
        self.check_intersection() # Check for intersection phase
        self.cleanup(object_ids) # Clean-up phase
        
    
    def cleanup(self, object_ids: list):
        # Object segments and exhausted ids clean-up section
        for object_segment_id in list(self.object_segments): # If object segment id is not in tracked objects id, clean it up
            if object_segment_id not in object_ids:
                del self.object_segments[object_segment_id]
        for exhausted_id in self.exhausted_ids: # If exhausted id is not in tracked objects id, clean it up
            if exhausted_id not in object_ids:
                self.exhausted_ids.remove(exhausted_id)

    
    def check_intersection(self):
        for object_id, object_segment in self.object_segments.items(): # Checks if current tracked object segments intersect line segment
            if object_id not in self.exhausted_ids and object_segment.intersect(self.line_segment): # If current tracked object segment intersects line segment and it is not exhausted, then intersection occurs
                self.count += 1
                self.exhausted_ids.append(object_id)
