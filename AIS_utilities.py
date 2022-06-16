# NPS utitilies for working with Automatic Information System (AIS)
# and canonical acoustic measurements in Glacier Bay National Park.

# geoprocessing libraries
import fiona
from fiona.crs import from_epsg
import pyproj
import geopandas as gpd
from shapely.ops import transform
from shapely.geometry import mapping, Point, Polygon
import rasterio
import rasterio.mask
from rasterio.plot import show

# some 'everyday' libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from functools import partial
import datetime as dt
import pytz
import os
import glob
import sys


# ======== utility functions =====================================

def get_utm_zone(longitude):
    
    return (int(1+(longitude+180.0)/6.0))

def spreadingLoss(Lw, distance, D=4.0, amb=35.0, amb_add=False):
    
    '''
    Standard equation to compute spreading loss from a point source.
    
    Inputs
    ------
    Lw: (float) sound power level of the point source in decibels
    distance: (float) or (array-like) distance from source in meters
    D: (float) default 4.0; the directionality of the source (4 = sphere, 2 = hemisphere)
    amb: (float) default 35.0; the natural ambient level at the reciever in decibels
    amb_add: (bool) default False; whether to return just source level at reciever, or source + ambience
    
    Return
    ------
    Lp: (float) or (array-like) the sound pressure level recieved with/without the influence of ambience
    '''
    
    Lp = Lw + 10*np.log10(1/(D*np.pi*np.power(distance,2)))
    
    if(amb_add):
        Lp = 10*np.log10(np.power(10, Lp/10)+np.power(10, amb/10))
    
    return Lp

def prepend(path, text):
    
    '''
    Function for adding header lines to files.
    '''
    
    with open(path, 'r+') as f:
        body = f.read()
        f.seek(0)
        f.write(text + body)
        
def point_buffer(lat, lon, km, return_equal_area=False):
    
    '''
    A generic circular buffer (with a given radius, `km`), around a coordinate.
    '''
    
    wgs84 = pyproj.CRS.from_epsg(4326)

    # Azimuthal equidistant projection
    aeqd_formatter = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    aeqd = pyproj.CRS.from_proj4(aeqd_formatter.format(lat=lat, lon=lon))

    # project the site coordinate into aeqd
    WGS84_to_AEQD = pyproj.Transformer.from_crs(wgs84, aeqd)
    long_m, lat_m = WGS84_to_AEQD.transform(lat, lon) # apply projection

    # to make a polygon: buffer point using a radius in meters
    buf_m = Point(long_m, lat_m).buffer(km * 1000)  # distance in meters

    if return_equal_area:
        buf = buf_m
    else:
        # convert polygon from aeqd back into wgs84
        AEQD_to_WGS84 = pyproj.Transformer.from_crs(aeqd, wgs84)
        buf = transform(AEQD_to_WGS84.transform, buf_m) # apply projection
    
    # somehow lat and long are incorrectly transposed... fix that
    buf = Polygon(np.flip(np.array(buf.exterior.xy), axis=0).T)

    return buf
        
# ======== project-specific functions =====================================

def load_MMSI_to_srcID():
    
    '''
    Helper function to load project-specific MMSI → srcID mapping
    
    Returns
    -------
    MMSI_to_srcID: (pd.DataFrame) containing mapping from one vessel identifier to another
    '''

    # all saved references will have the following format
    potential_files = glob.glob(r"T:\ResMgmt\WAGS\Sound\Experiments & Data\2020 GLBA Acoustic Inventory\MMSI_to_srcID*.csv")

    # split apart to retrieve the date string suffix
    potential_dates = [os.path.basename(f).split("_")[-1][:-4] for f in potential_files]

    # find the time elapsed between the save suffix and today
    time_elapsed_since_save = []
    for pdate in potential_dates:
        
        try:
            file_date = dt.datetime.strptime(pdate, "%Y-%m-%d")
            time_elapsed_since_save.append(dt.datetime.now() - file_date)
            
        except ValueError:
            time_elapsed_since_save.append(dt.timedelta(days=10000))
        
    history = np.array(time_elapsed_since_save)

    most_recent_file = potential_files[np.argwhere(history == history.min())[0][0]]
    
    MMSI_to_srcID = pd.read_csv(most_recent_file, index_col="MMSI")
    
    return MMSI_to_srcID

def satellite_basemap(mask):
    
    '''
    Convenient satellite imagery basemap
    '''

    # Compressed Glacier Bay National Park and Preserve Satellite Imagery 
    # Mosaic bands 321 Pan Sharpened
    rasterPath = r"C:\Users\DBetchkal\Documents\ArcGIS\Projects\GLBA_AIS_20200925\GLBAPansharp.tif"

    # create a geoDataFrame to contain the buffer
    gdf_buffer = gpd.GeoSeries([mask], crs="EPSG:4326")
    
    # load the satellite image raster
    satellite = rasterio.open(rasterPath)
    
    # mask the satellite image using the site buffer
    out_image, out_transform = rasterio.mask.mask(satellite, gdf_buffer, crop=True)

    return out_image, out_transform


def create_circular_site_buffer(ds, unit, site, year, search_within_km = 20):
    
    '''
    Using AKR metadata, create a circular buffer around an existing acoustic monitoring site.
    '''
    
    # load the metadata sheet
    metadata = pd.read_csv(r"V:\Complete_Metadata_AKR_2001-2021.txt", delimiter="\t", encoding = "ISO-8859-1")

    # look up the site's coordinates in WGS84
    lat_in, long_in = metadata.loc[(metadata["code"] == site)&(metadata["year"] == year), "lat":"long"].values[0]
    
    # create the buffer polygon around the site
    buf = point_buffer(lat_in, long_in, search_within_km)
    
    return buf


def track_to_station_distances(track_data, site_point):
    
    '''
    For a geopandas dataframe containing lat/long columns,
    find the distances between each point and a site coordinate.
    
    Inputs
    ------
    
    track_data (geodataframe): must contain columns "LatitudeWGS84" and "LongitudeWGS84"
    site_point (numpy array): an array of shape (2,) representing [latitude, longitude] in WGS84
    
    Returns
    -------
    
    distances (numpy array): distance in meters between each point and site
    
    '''
    
    # unpack incoming site data
    lat_site_in, long_site_in = site_point
    
    # lookup the UTM zone using the first point
    zone = get_utm_zone(long_site_in)

    # epsg codes for Alaskan UTM zones
    epsg_lookup = {1:'epsg:26901', 2:'epsg:26902', 3:'epsg:26903', 4:'epsg:26904', 5:'epsg:26905', 
                   6:'epsg:26906', 7:'epsg:26907', 8:'epsg:26908', 9:'epsg:26909', 10:'epsg:26910'}

    # convert from D.d (WGS84) to meters (NAD83)
    in_proj = pyproj.CRS.from_string('epsg:4326')
    out_proj = pyproj.CRS.from_string(epsg_lookup[zone])

    # create a Transformer object
    WGS84_to_NAD83 = pyproj.Transformer.from_crs(in_proj, out_proj)
    
    # transform the site's coordinates
    lat_site_out, long_site_out = WGS84_to_NAD83.transform(lat_site_in, long_site_in) # apply projection
    
    distances = []
    for meta, row in track_data.iterrows():
        
        # unpack the incoming gps data
        lat_in = row["LatitudeWGS84"]
        long_in = row["LongitudeWGS84"]

        # transform the current vessel coordinate
        lat_out, long_out = WGS84_to_NAD83.transform(lat_in, long_in) # apply projection

        distance = np.linalg.norm(np.array([lat_out, long_out])-np.array([lat_site_out, long_site_out]))
        distances.append(distance)
    
    return np.array(distances)


def tracks_within_acoustic_record(ds, unit, site, year, AIS_points):
    
    # ===== first part; site coordinate wrangling =====================
    
    # load the metadata sheet
    metadata = pd.read_csv(r"V:\Complete_Metadata_AKR_2001-2021.txt", delimiter="\t", encoding = "ISO-8859-1")

    # look up the site's coordinates in WGS84
    lat_in, long_in = metadata.loc[(metadata["code"] == site)&(metadata["year"] == year), "lat":"long"].values[0]

    # lookup the UTM zone using the first point
    zone = get_utm_zone(long_in)

    # epsg codes for Alaskan UTM zones
    epsg_lookup = {1:'epsg:26901', 2:'epsg:26902', 3:'epsg:26903', 4:'epsg:26904', 5:'epsg:26905', 
                   6:'epsg:26906', 7:'epsg:26907', 8:'epsg:26908', 9:'epsg:26909', 10:'epsg:26910'}

    # convert from D.d (WGS84) to meters (NAD83)
    in_proj = pyproj.CRS.from_string('epsg:4326')
    out_proj = pyproj.CRS.from_string(epsg_lookup[zone])

    # convert into NMSIM's coordinate system
    WGS84_to_NAD83 = pyproj.Transformer.from_crs(in_proj, out_proj)
    lat_out, long_out = WGS84_to_NAD83.transform(lat_in, long_in) # apply projection
    
    # ===== second part; mic height to feet =====================

    # look up the site's coordinates in WGS84
    height = metadata.loc[(metadata["code"] == site)&(metadata["year"] == year), "microphone_height"].values[0]

    print(unit+site+str(year)+":", "{0:.0f},".format(long_out), "{0:.0f}".format(lat_out), "- UTM zone", zone)
    print("\tmicrophone height", "{0:.2f} feet.".format(height*3.28084),"\n")

    # ===== third part; determine the date range of NVSPL files ===============

    # load the datetime of every NVSPL file
    NVSPL_dts = pd.Series([dt.datetime(year=int(e.year),
                                       month=int(e.month),
                                       day=int(e.day),
                                       hour=int(e.hour),
                                       minute=0,
                                       second=0) for e in ds.nvspl(unit=unit, site=site, year=year)])

    # everything should be in chronological order, but just in case...
    NVSPL_dts.sort_values(inplace=True, ascending=True)

    # retrieve the start/end bounds and convert back to YYYY-MM-DD strings
    start, end = (dt.datetime.strftime(d, "%Y-%m-%d") for d in [NVSPL_dts.iloc[0], NVSPL_dts.iloc[-1]])
    print("\n\tRecord begins", start, "and ends", end, "\n")
    
    # filter out only the events that overlap the sampling range
    filtered_tracks = AIS_points.loc[((AIS_points["Datetime"] >= NVSPL_dts.iloc[0])&
                                      (AIS_points["Datetime"] < NVSPL_dts.iloc[-1])), :]
    
    # as a precaution, sort by Datetime
    filtered_tracks = filtered_tracks.sort_values("Datetime")
    
    return filtered_tracks, lat_in, long_in


def overview_map(data, MMSI, site, year, buffer, long, lat, savePath):
    
    try:
    
        # for this MMSI, plot all the Events using different colors
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='white')

        # load the satellite imagery basemap and add it to the map frame
        im, transformer = satellite_basemap(buffer)
        rasterio.plot.show(2*im, ax=ax, transform=transformer)

        # add the site as a bright green point
        ax.plot(long, lat, ls="", marker="o", markersize=7, zorder=100, 
                  color="lime", label=site)

        # add the vessel tracks
        _ = data.plot(column="Events", ax=ax, ls="--", lw=0.5, markersize=1, cmap="tab20")

        # give the map title and axis labels
        ax.set_ylabel("Latitude", labelpad=15, fontsize=9)
        ax.set_xlabel("Longitude", labelpad=15, fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=9)

        # this will create an approximately square plot
        ax.set_aspect(aspect=2)

        # move the map all the way to the left
        ax.set_anchor('W')

        # draw the map legend
        ax.legend()

        # the figure's supertitle includes MMSI, vessel name (if available) and "event number" for easy lookup later
        if(data["ShipName"].iloc[0].title() is not None):
            plt.title("MMSI "+str(MMSI)+" - "+data["ShipName"].iloc[0].title(), 
                          y=1.00, fontsize=13, loc="left")
        else:
            plt.title("MMSI "+str(MMSI)+" - "+data["ShipName"].iloc[0].title(), 
                          y=1.00, fontsize=13, loc="left")

        plt.xticks(rotation=45)
        plt.savefig(savePath + os.sep + "overviews\GLBA"+site+str(year)+"_MMSI_"+str(MMSI)+"_eventOverview.png",
                dpi=200, bbox_inches="tight")

        plt.show()
        plt.close()
        
    except AttributeError:
        print("Encounted an Attribute Error while creating the overview map... skipping.")
     
    
def contiguous_regions(condition):

    """
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)

    return idx


def load_AIS_from_gdb(gdb_path, mask=None):
    
    '''
    Parsing function for pre-compiled AIS data (*.gdb)
    
    gdb_path: (str) path to *.gdb
    mask: (gpd.GeoDataFrame) optional default None; polygonal mask to spatially filter the raw dataset
    '''
    
    layerMap = {"aircraft":0, "vessel":1}
    
    # load the tracks from Whitney's geodatabase
    AIS = gpd.read_file(gdb_path, layer=layerMap["vessel"], mask=mask) 
    
    # add a new column containing a Python datetime
    AIS["Datetime"] = AIS["BaseStationTimeUTC"].apply(lambda t: dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S+00:00"))
    
    return AIS


def AIS_filter_from_acoustic_record(unit, site, year, acoustic_ds, AIS_ds):
    
    '''
    Select only AIS files which overlap the acoustic record.
    
    inputs
    ------
    unit: (str) 4-letter park unit alpha code (i.e., "GLBA")
    site: (str) deployment alpha-numeric code (i.e., "002", "HUTCH", etc...)
    year: (int) 4-digit year (i.e., "2020")
    
    returns
    -------
    flat_paths: (list) paths for all AIS *.csv files which overlap the acoustic record
    
    '''
    
    unique_dates = pd.DataFrame(set([(e.year, e.month, e.day) for e in acoustic_ds.nvspl(unit=unit, site=site, year=year)]),
                                columns=["Year", "Month", "Day"])

    paths = []
    for _, row in unique_dates.iterrows():
        
        # find the unique unit+year+month+day AIS file, if any exists
        paths.append([e.path for e in AIS_ds.AIS(unit=unit,
                                                 year=int(row.Year), 
                                                 month=int(row.Month), 
                                                 day=int(row.Day))])

    # because some days have multiple AIS files, list lengths are ≥1
    flat_paths = [item for sublist in paths for item in sublist]

    return flat_paths


def load_AIS_from_csv(csv_path, n=None, column_convention="raw", mask=None, iyore_subset=None):
    
    '''
    Parsing function for raw AIS data from the Alaska Marine Exchange (*.csv)
    
    Inputs
    ------
    csv_path: (str) path to *.csv folder; currently does not handle recursive searching
    n: (int) or default None; the number of .csv files to load
    column_convention: (str) default "raw", but also accepts "GLBA" to use park convention
    mask: (gpd.GeoDataFrame) optional default None; polygonal mask to spatially filter the raw dataset
    iyore_subset: (list) optional default None; subset list of *.csv files
    
    Returns
    -------
    AIS: (gpd.GeoDataFrame) formatted Automatic Information System data with point geometry
    '''
    
    import os
    import glob
        
    if iyore_subset is not None:
        raws = iyore_subset
    else:
        # load in all the raw CSV files
        raws = glob.glob(csv_path + os.sep + "*.csv")

    if n is not None:
        raws = raws[:n]

    print("\t\t"+str(len(raws))+" .csv files will be processed...")
    
    # glean the date that each file represents (necessary?)
    dates = [dt.datetime.strptime(r.split("-")[-2], "%Y%m%d") for r in raws]

    print("\t\tDates have been stripped from the files...")

    # concatenate all the CSV files together into one `pandas` dataframe
    AIS_df = pd.concat([pd.read_csv(r, low_memory=False) for r in raws])
    
    print("\t\tFiles have been concatenated into a `pd.DataFrame` object...")

    # there's a pile of 1090 MHz jet ADS-B points in this dataset, too
    # DROP THEM!
    AIS_df = AIS_df.loc[pd.notna(AIS_df['Ship name']), :]
    
    print("\t\t1090 MHz ADS-B data have been dropped...")

    if column_convention == "GLBA":
        
        # a nice AIS column-name convention from Whitney Rapp
        AIS_df.columns = ['BaseStationTimeUTC', 'MMSI', 'IMO', 'ShipName', 'TypeOfShip',
                          'SizeA_m', 'SizeB_m', 'SizeC_m', 'SizeD_m', 'Draught_m', 'SOG_kt', 
                          'COG', 'HeadingTrue', 'LatitudeWGS84', 'LongitudeWGS84', 'Destination', 
                          'NavigationalStatus', 'Country', 'TargetClass', 'DataSourceType', 
                          'DataSourceRegion']
        
        # set up the geometry
        geo = gpd.points_from_xy(AIS_df.LongitudeWGS84, AIS_df.LatitudeWGS84)
        
        # add a new column containing a Python datetime
        AIS_df["Datetime"] = AIS_df["BaseStationTimeUTC"].apply(lambda t: dt.datetime.strptime(t, '%d %b %Y %H:%M:%S UTC'))
    
        
    elif column_convention == "raw":
        
        # columns "as is", but we'll still set up the geometry
        geo = gpd.points_from_xy(AIS_df.Longitude, AIS_df.Latitude)
        
        try:
            # add a new column containing a Python datetime
            AIS_df["Datetime"] = AIS_df['Base station time stamp'].apply(lambda t: dt.datetime.strptime(t, '%d %b %Y %H:%M:%S UTC'))
        
        except ValueError:

            # add a new column containing a Python datetime
            AIS_df["Datetime"] = AIS_df['Base station time stamp'].apply(lambda t: dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S UTC'))


    # then use the 'Latitude' and 'Longitude' columns to convert into a geopandas `GeoDataFrame` object
    AIS = gpd.GeoDataFrame(AIS_df, geometry=geo, crs='EPSG:4326')
    
    # this is necessary before clipping
    AIS.reset_index(inplace=True)

    print("\t\tRaw `gpd.GeoDataFrame` object has been created...")
    
    # spatially filter using mask if provided
    if mask is not None:
        
        mask_gdf = gpd.GeoDataFrame(index=["buffer"], geometry=[mask], crs='EPSG:4326')
        AIS = gpd.clip(AIS, mask_gdf)
        
    else:
        
        pass
    
    print("\t\tFinished creating GeoDataFrame!")
    return AIS


def geometric_nfi(gdf_tracks):
    
    '''
    Find the time periods where AIS-equipped vessels
    are not within the buffered area. (Or their signals
    are not being broadcasted...)
    
    Returns
    -------
    NFIs (numpy array): length of periods without vessels (in seconds)
    '''
    
    # create an array for subsequent (geometric) NFI analysis
    NFI_pairs = np.array([])

    # iterate through each vessel
    for MMSI, data in gdf_tracks.groupby("MMSI"):

        # this 'one-liner' sorts the datetimes, differences them, converts the timedeltas to minutes, and binarizes
        # into a boolean (gaps > 15 minutes: True, gaps ≤ 15 minutes: False)
        # since this is an analysis of Noise-Free Intervals
        data["Events"] = data["Datetime"].sort_values().diff().apply(lambda t: t.total_seconds()/60) > 15.0

        # a concise way to group by temporal gaps; overwriting the previous boolean
        data["Events"] = data["Events"].apply(lambda x: 1 if x else 0).cumsum()
        data["Events"] = data["Events"]+1 # overcome zero-indexing

        # our goal is to summarize noise by event
        for event, event_data in data.groupby("Events"):

            # use the entry point to compute the UTC offset
            offset = akt.utcoffset(event_data["Datetime"].iloc[0])

            # when did the event enter/exit the buffered area?
            enter = event_data["Datetime"].iloc[0] + offset
            exit = event_data["Datetime"].iloc[-1] + offset

            # append array with enter/exit times
            NFI_pairs = np.append(NFI_pairs, [enter.to_pydatetime(), 
                                              exit.to_pydatetime()])


    # group the pairs by event
    NFI_pairs = NFI_pairs.reshape((int(NFI_pairs.shape[0]/2), 2))

    # begin and end times as datetime objects
    time_s = gdf_tracks["Datetime"].min().to_pydatetime()
    time_e = gdf_tracks["Datetime"].max().to_pydatetime()

    # an array containing every second of the record as datetime
    all_seconds = np.array([time_s + dt.timedelta(seconds=i) for 
                            i in range(int((time_e-time_s).total_seconds()))])

    # a boolean array the same length as the record
    within_times = np.zeros(all_seconds.shape)

    for NFI_pair in NFI_pairs:

        try:

            # convert the enter/exit times into integer indices
            ind_begin = np.argwhere(all_seconds == NFI_pair[0])[0][0]
            ind_end = np.argwhere(all_seconds == NFI_pair[1])[0][0]

            # set values between the two indices equal to 'True'
            within_times[ind_begin:ind_end] = 1.0

        except IndexError:
            pass # basically the event had no duration


    # finally, find the times where vessels were not in the buffer
    # and do a bit of rearranging to arrive at (geometric) NFI values
    nfi_bounds = contiguous_regions(within_times == 0)
    NFIs = (nfi_bounds.T[1] - nfi_bounds.T[0])

    return NFIs

        
def SPLAT_row_dictionary(nvsplDate, hr, start_secs, len_secs, srcID=3):
    
    '''
    Return a SRCID row-dictionary
    a row-by-row list of these dictionaries can be used
    to initialize a SRCID file as `pd.DataFrame`
    
    Inputs
    ------
    nvsplDate: (str) date of event as "YYYY-MM-DD"
    hr: (int) hour the event occurred, using 24-hour time
    start_secs: (int) the second of the hour at which the event started
    len_secs: (int) the duration of the event in seconds
    
    Returns
    -------
    dictionary: (dict) one complete row of a SRCID file
    
    '''

    dictionary = {'nvsplDate':nvsplDate,
                  'hr':hr,
                  'secs':start_secs,
                  'len':len_secs,
                  'srcID':srcID, 
                  'Hz_L':12.5,
                  'Hz_U':2000,
                  'MaxSPL':0,
                  'SEL':0,
                  'MaxSPLt':0,
                  'SELt':0,
                  'userName':'AIS_derived',
                  'tagDate': dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d %H:%M:%S")}

    return dictionary


def presence_bounds_to_SRCID_bounds(SRCID_row_list, srcID, enter, exit):

    # round down start hour, round up end hour
    start_bound = enter.replace(second=0, microsecond=0, minute=0, hour=enter.hour)
    end_bound = exit.replace(second=0, microsecond=0, minute=0, hour=exit.hour) + dt.timedelta(hours=1)

    # determine the timespan (in hours) and generate a list of all hours elapsed
    span = int((end_bound - start_bound).total_seconds()/3600)
    hrs = [start_bound + dt.timedelta(hours=int(hr_step)) for hr_step in np.arange(span)]

    for k, hr in enumerate(hrs):

        if(k == 0): # starting case

            start_secs = (enter.minute*60 + enter.second)
            len_secs = 3600 - start_secs
            out_date = dt.datetime.strftime(hr.date(), "%Y-%m-%d")
            SRCID_row_list.append(SPLAT_row_dictionary(out_date, hr.hour, start_secs, len_secs, srcID))

        elif(k == span - 1): # ending case

            start_secs = 0
            len_secs = (exit.minute*60 + exit.second)
            out_date = dt.datetime.strftime(hr.date(), "%Y-%m-%d")
            SRCID_row_list.append(SPLAT_row_dictionary(out_date, hr.hour, start_secs, len_secs, srcID))

        else: # all middle cases

            start_secs = 0
            len_secs = 3600
            out_date = dt.datetime.strftime(hr.date(), "%Y-%m-%d")
            SRCID_row_list.append(SPLAT_row_dictionary(out_date, hr.hour, start_secs, len_secs, srcID))

    return SRCID_row_list


def create_SRCID_from_AIS(site, AIS_filtered, savePath):

    # what will become a list of dictionaries
    SRCID_raw = []

    # our local time zone
    akt = pytz.timezone('US/Alaska')

    # create an array to hold start/end pairs for geometric NFI estimates
    NFI_pairs = np.array([])

    # load mapping MMSI → srcID
    MMSI_to_srcID = load_MMSI_to_srcID()

    # iterate through each vessel
    for MMSI, data in AIS_filtered.groupby("MMSI"):

        print("================ now working on MMSI", MMSI, "================ \n")

        # this 'one-liner' sorts the datetimes, differences them, converts the timedeltas to minutes, and binarizes
        # into a boolean (gaps > 15 minutes: True, gaps ≤ 15 minutes: False)
        data["Events"] = data["Datetime"].sort_values().diff().apply(lambda t: t.total_seconds()/60) > 15.0

        # a concise way to group by temporal gaps; overwriting the previous boolean
        data["Events"] = data["Events"].apply(lambda x: 1 if x else 0).cumsum()
        data["Events"] = data["Events"]+1 # overcome zero-indexing

        # our goal is to summarize noise by event
        for event, event_data in data.groupby("Events"):

            current_srcID = "{:.3f}".format(MMSI_to_srcID.loc[MMSI, "srcID"])

            print("MMSI", MMSI, "event #"+str(event))
            print("this will be srcID", current_srcID)

            # use the entry point to compute the UTC offset
            offset = akt.utcoffset(event_data["Datetime"].iloc[0])

            # when did the event enter/exit the buffered area?
            enter = event_data["Datetime"].iloc[0] + offset
            exit = event_data["Datetime"].iloc[-1] + offset
            duration = exit - enter

            # check to see if duration is non-zero
            if(duration <= dt.timedelta(seconds=0)):
                pass

            elif(duration > dt.timedelta(seconds=0)):

                print("Alaska [enter/exit]:", enter, "/", exit)
                print("Vehicle within radius for", exit-enter)
                print("")

                # append array with enter/exit times
                NFI_pairs = np.append(NFI_pairs, [enter, exit])

                # append SRCID dictionaries to the existing list
                SRCID_raw = presence_bounds_to_SRCID_bounds(SRCID_raw, current_srcID, enter, exit)


    SRCID = pd.DataFrame(SRCID_raw)

    #sort the dataframe by date/time
    SRCID = SRCID.sort_values("nvsplDate")

    # export to csv with correct naming convention based on station code
    SRCID.to_csv(savePath + os.sep + "SRCID_GLBA" + site + ".txt", sep = "\t",index = False )

    # add the splat format version header to the file
    prepend(savePath + os.sep + "SRCID_GLBA" + site +".txt", "%% SRCID file v20111005" + "\n")