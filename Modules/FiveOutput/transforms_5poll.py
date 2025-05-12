import os

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import random
import numpy as np
import torch
from rasterio.plot import reshape_as_image

# define image transforms
class ChangeBandOrder(object):
    def __call__(self, sample):
        """necessary if model was pre-trained on .npy files of BigEarthNet and should be used on other Sentinel-2 images

        move the channels of a sentinel2 image such that the bands are ordered as in the BigEarthNet dataset
        input image is expected to be of shape (200,200,12) with band order:
        ['B04', 'B03', 'B02', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09'] (i.e. like my script on compute01 produces)

        output is of shape (12,120,120) with band order:
        ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] (order in BigEarthNet .npy files)
        """
        img = sample["img"].copy()
        img = np.moveaxis(img, -1, 0)
        reordered_img = np.zeros(img.shape)
        reordered_img[0, :, :] = img[10, :, :]
        reordered_img[1, :, :] = img[2, :, :]
        reordered_img[2, :, :] = img[1, :, :]
        reordered_img[3, :, :] = img[0, :, :]
        reordered_img[4, :, :] = img[4, :, :]
        reordered_img[5, :, :] = img[5, :, :]
        reordered_img[6, :, :] = img[6, :, :]
        reordered_img[7, :, :] = img[3, :, :]
        reordered_img[8, :, :] = img[7, :, :]
        reordered_img[9, :, :] = img[11, :, :]
        reordered_img[10, :, :] = img[8, :, :]
        reordered_img[11, :, :] = img[9, :, :]

        if img.shape[1] != 120 or img.shape[2] != 120:
            reordered_img = reordered_img[:, 40:160, 40:160]

        out = {}
        for k,v in sample.items():
            if k == "img":
                out[k] = reordered_img
            else:
                out[k] = v

        return out

class ToTensor(object):
    def __call__(self, sample):
        img = torch.from_numpy(sample["img"].copy())

        if sample.get("no2") is not None:
            no2 = torch.from_numpy(sample["no2"].copy())
        if sample.get("o3") is not None:
            o3 = torch.from_numpy(sample["o3"].copy())
        if sample.get("co") is not None:
            co = torch.from_numpy(sample["co"].copy())
        if sample.get("so2") is not None:
            so2 = torch.from_numpy(sample["so2"].copy())
        if sample.get("pm10") is not None:
            pm10 = torch.from_numpy(sample["pm10"].copy())

        if sample.get("s5p") is not None:
            s5p = torch.from_numpy(sample["s5p"].copy())

        out = {}
        for k,v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "no2":
                out[k] = no2
            elif k == "o3":
                out[k] = o3
            elif k == "co":
                out[k] = co
            elif k == "so2":
                out[k] = so2
            elif k == "pm10":
                out[k] = pm10
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v

        return out

class DatasetStatistics(object):
    def __init__(self):
        self.channel_means = np.array([340.76769064, 429.9430203, 614.21682446,
                590.23569706, 950.68368468, 1792.46290469, 2075.46795189, 2218.94553375,
                2266.46036911, 2246.0605464, 1594.42694882, 1009.32729131])

        self.channel_std = np.array([554.81258967, 572.41639287, 582.87945694,
                675.88746967, 729.89827633, 1096.01480586, 1273.45393088, 1365.45589904,
                1356.13789355, 1302.3292881, 1079.19066363, 818.86747235])

        # statistics over the whole of Europe from Sentinel-5P products in 2018-2020:
        # l3_mean_europe_2018_2020_005dg.netcdf mean 1.51449095e+15 std 6.93302798e+14
        # l3_mean_europe_large_2018_2020_005dg.netcdf mean 1.23185273e+15 std 7.51052046e+14
        self.s5p_mean = 1.23185273e+15      #this will need to be updated if I change the samples file!!!
        self.s5p_std = 7.51052046e+14       #this will need to be updated if I change the samples file!!!

        #altitude values
        self.alt_mean = 246.9503722
        self.alt_std = 255.961989

        # #popdense values
        # self.popdense_mean = 435.719603
        # self.popdense_std = 1049.972678

        # values for averages from 2018-2020 per EEA station, across stations
        # values for averages from 2018-2020 per EEA station, across stations
        self.no2_mean = 15.882053 #updated to 3poll dataset
        self.no2_std =  9.899851 #updated to 3poll dataset

        self.o3_mean = 54.490576	 #updated to 3poll dataset
        self.o3_std = 18.109822	 #updated to 3poll dataset

        self.co_mean = 0.308071
        self.co_std = 0.190605	

        self.so2_mean = 4.038745
        self.so2_std = 3.784285

        self.pm10_mean = 20.313453 
        self.pm10_std = 9.335500 

        # add 
        self.tavg_mean = 14.413344 
        self.tavg_std =  6.851507 

        self.prcp_mean = 68.462932 
        self.prcp_std = 74.003820 

        self.wspd_mean = 11.033690	
        self.wspd_std = 3.396612

        self.pres_mean = 1016.292834	
        self.pres_std = 4.148925	

        self.tsun_mean = 12108.385502 
        self.tsun_std = 5542.400919         
        # end add

class Normalize(object):
    """normalize a sample, i.e. the image and NO2 value, by subtracting mean and dividing by std"""
    def __init__(self, statistics):
        self.statistics = statistics

    def __call__(self, sample):
        img = reshape_as_image(sample.get("img").copy())
        img = np.moveaxis((img - self.statistics.channel_means) / self.statistics.channel_std, -1, 0)

        if sample.get("no2") is not None:
            no2 = sample.get("no2")#.copy()
            no2 = np.array((no2 - self.statistics.no2_mean) / self.statistics.no2_std)
            #no2 = np.array((no2 - 0) / 1)

        if sample.get("o3") is not None:
            o3 = sample.get("o3")  # .copy()
            o3 = np.array((o3 - self.statistics.o3_mean) / self.statistics.o3_std)
            #o3 = np.array((o3 - 0) / 1)

        if sample.get("co") is not None:
            co = sample.get("co")  # .copy()
            co = np.array((co - self.statistics.co_mean) / self.statistics.co_std)
            #co = np.array((co - 0) / 1)

        if sample.get("so2") is not None:
            so2 = sample.get("so2")  # .copy()
            so2 = np.array((so2 - self.statistics.so2_mean) / self.statistics.so2_std)
            #so2 = np.array((so2 - 0) / 1)

        if sample.get("pm10") is not None:
            pm10 = sample.get("pm10")  # .copy()
            pm10 = np.array((pm10 - self.statistics.pm10_mean) / self.statistics.pm10_std)
            #pm10 = np.array((pm10 - 0) / 1)

        if sample.get("s5p") is not None:
            #print("Sentinel5p online")
            #print(sample.get("s5p"))
            s5p = sample.get("s5p").copy()
            s5p = np.array((s5p - self.statistics.s5p_mean) / self.statistics.s5p_std)

        if sample.get("Altitude") is not None:
            alt = sample.get("Altitude")#.copy()
            alt = np.array((alt - self.statistics.alt_mean) / self.statistics.alt_std)

        if sample.get("PopulationDensity") is not None:
            PopulationDensity = sample.get("PopulationDensity")#.copy()
            PopulationDensity = np.array((PopulationDensity - self.statistics.popdense_mean) / self.statistics.popdense_std)

        # add
        if sample.get("tavg") is not None:
            tavg = sample.get("tavg")#.copy()
            tavg = np.array((tavg - self.statistics.tavg_mean) / self.statistics.tavg_std)

        if sample.get("prcp") is not None:
            prcp = sample.get("prcp")#.copy()
            prcp = np.array((prcp - self.statistics.prcp_mean) / self.statistics.prcp_std)

        if sample.get("wspd") is not None: 
            wspd = sample.get("wspd")#.copy()
            wspd = np.array((wspd - self.statistics.wspd_mean) / self.statistics.wspd_std)

        if sample.get("pres") is not None:
            pres = sample.get("pres")#.copy()
            pres = np.array((pres - self.statistics.pres_mean) / self.statistics.pres_std)

        if sample.get("tsun") is not None:
            tsun = sample.get("tsun")#.copy()
            tsun = np.array((tsun - self.statistics.tsun_mean) / self.statistics.tsun_std)
        # end

        out = {}
        for k,v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "no2":
                out[k] = no2
            elif k == "o3":
                out[k] = o3
            elif k == "co":
                out[k] = co
            elif k == "so2":
                out[k] = so2
            elif k == "pm10":
                out[k] = pm10
            elif k == "s5p":
                out[k] = s5p
            elif k == "Altitude":
                out[k] = alt
            # elif k == "PopulationDensity":
            #     out[k] = PopulationDensity

            # add
            elif k == "tavg":
                out[k] = tavg
            elif k == "prcp":
                out[k] = prcp
            elif k == "wspd":
                out[k] = wspd
            elif k == "pres":
                out[k] = pres
            elif k == "tsun":
                out[k] = tsun
            # end add

            else:
                out[k] = v

        return out

    @staticmethod
    def undo_no2_standardization(statistics, no2):
        return (no2 * statistics.no2_std) + statistics.no2_mean
    @staticmethod
    def undo_o3_standardization(statistics, o3):
        return (o3 * statistics.o3_std) + statistics.o3_mean
    @staticmethod
    def undo_co_standardization(statistics, co):
        return (co * statistics.co_std) + statistics.co_mean
    @staticmethod
    def undo_so2_standardization(statistics, so2):
        return (so2 * statistics.so2_std) + statistics.so2_mean
    @staticmethod
    def undo_pm10_standardization(statistics, pm10):
        return (pm10 * statistics.pm10_std) + statistics.pm10_mean

class Randomize():
    def __call__(self, sample):
        img = sample.get("img").copy()

        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = sample["s5p"].copy()

        if random.random() > 0.5:
            img = np.flip(img, 1)
            if s5p_available: s5p = np.flip(s5p, 0)
        if random.random() > 0.5:
            img = np.flip(img, 2)
            if s5p_available: s5p = np.flip(s5p, 1)
        if random.random() > 0.5:
            img = np.rot90(img, np.random.randint(0, 4), axes=(1,2))
            if s5p_available: s5p = np.rot90(s5p, np.random.randint(0, 4), axes=(0,1))

        out = {}
        for k,v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v

        return out