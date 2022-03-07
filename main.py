import argparse
from bin import counter, detector, geometry
import cv2
import norfair
import numpy as np
from norfair import Tracker, Video


max_distance_between_points: int = 30


def euclidean_distance(detection, tracked_object):
    """
    Computes the euclidean distance between detections and tracked estimations
    """
    return np.linalg.norm(detection.points - tracked_object.estimate)


def list_camera_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working
    """
    non_working_ports = []
    working_ports = []
    available_ports = []
    dev_port = 0
    while len(non_working_ports) <= 3: # If there are more than 3 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port, w, h))
                working_ports.append(dev_port)
            else:
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports, working_ports, non_working_ports


def main(args):
    # Instantiates the YOLO model
    model = detector.YOLO(args.detector_path, device=args.device)
    
    # Handles the camera ports automatically
    _, camera_ports, _ = list_camera_ports()
    if args.camera_id is None and camera_ports != []: # camera_id is not specified
        camera = camera_ports[0]
        video = Video(camera)        
    elif args.camera_id in camera_ports: # specified camera_id is working
        camera = camera_ports[args.camera_id]
        video = Video(camera)
    else: # specified camera_id is not working
        raise Exception(
            "Selected camera id is either wrong or not working"
            )
    
    # TODO: Handle multiple trackers for multiple classes!
    # Instantiates the tracker
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
        )
    # Defines the counter virtual line segment according to the orientation
    width = int(video.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.counter_orientation == 'vertical':
        first_endpoint = geometry.Point(int(width/2), 0)
        second_endpoint = geometry.Point(int(width/2), height)
    elif args.counter_orientation == 'horizontal':
        first_endpoint = geometry.Point(0, int(height/2))
        second_endpoint = geometry.Point(width, int(height/2))
    else:
        raise Exception(
            "Selected orientation is not correct"
            )
    line_segment = geometry.Segment(first_endpoint, second_endpoint)
    counter_class = counter.Counter(
        line_segment
        )
    print("Press ESC key to quit at any time")
    
    while True:
        ret, frame = video.video_capture.read()
        if ret is False or frame is None:
            break
        # Performs the detections over the current frame
        # Detection is structured as follows: https://github.com/ultralytics/yolov5/blob/9ccfa85249a2409d311bdf2e817f99377e135091/models/common.py#L295-L309
            # .names: class names 
            # .xywh: list of tensors xywh[0] = (xywh, conf, cls) 
        # TODO: Alternatively, handle multiple detections for multiple classes!
        yolo_detections = model(
            frame,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thresh,
            image_size=args.img_size,
            classes=args.classes
            )
        detections = detector.yolo_detections_to_norfair_detections(yolo_detections, track_points=args.track_points)
        # Updates the tracked objects locations
        tracked_objects = tracker.update(detections=detections)
        counter_class.update(tracked_objects)
        cv2.line(frame, first_endpoint.to_tuple(), second_endpoint.to_tuple(), (0, 0, 255), 5)
        # TODO: Evaulate showing count on screen or store in memory
        # Count debug print
        try:
            print(counter_class.count)
        except:
            pass
        # Shows either the bbox or centroids for each tracked object
        if args.track_points == 'centroid':
            norfair.draw_points(frame, detections)
        elif args.track_points == 'bbox':
            norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
        
        cv2.imshow('Object tracking', frame)
        k = cv2.waitKey(1)
        if k == 27: # Press ESC key to quit
            video.video_capture.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tracks and counts objects within a camera stream")
    parser.add_argument("--camera_id", type=int, default=None, help="Camera device port to use")
    parser.add_argument("--detector_path", type=str, default="yolov5m6.pt", help="YOLOv5 model path")
    parser.add_argument("--img_size", type=int, default="720", help="YOLOv5 inference size (pixels)")
    parser.add_argument("--conf_thres", type=float, default="0.25", help="YOLOv5 object confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='Filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument("--counter_orientation", type=str, default="vertical", help="Counter virtual line segment orientation: 'vertical' or 'horizontal'")
    parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
    parser.add_argument("--track_points", type=str, default="bbox", help="Draw tracked points as: 'centroid' or 'bbox'")
    args = parser.parse_args()
    main(args)