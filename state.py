import enum


class State(enum.Enum):
    Unknown = 0
    Init = 1
    Standby = 2
    InformationCapture = 3
    MotionTracking = 4

