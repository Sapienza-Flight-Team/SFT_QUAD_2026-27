import ctypes as ct

TEXT_CAPACITY = 64

FIELD_REGISTRY = {
    "PhotoExif": {
        "frame_counter": ct.c_uint64,
        "exposure_time": ct.c_double,
        "gain": ct.c_double,
        "black_level": ct.c_uint32,
        "bsl_color_space": ct.c_char * TEXT_CAPACITY,
        "gamma": ct.c_uint32,
        "light_source_preset": ct.c_char * TEXT_CAPACITY,
   },
   "PixData": {
        "lat": ct.c_double,
        "lon": ct.c_double,
        "alt_msl": ct.c_double,
        "alt_rel": ct.c_double,
        "roll": ct.c_double,
        "pitch": ct.c_double,
        "yaw": ct.c_double,
        "camera_trigger_seq": ct.c_int64,
        "mission_current_wp": ct.c_uint32,
        "mission_total_wp": ct.c_uint32,
   } 
}

biglist = [(field, fieldsize) for classtype in FIELD_REGISTRY.values() for field, fieldsize in classtype.items()]

print(biglist)