# -*- coding: utf-8 -*-

def classFactory(iface):
    from .hyperspatial_bigdata import HyperSpatialBigData
    return HyperSpatialBigData(iface)

