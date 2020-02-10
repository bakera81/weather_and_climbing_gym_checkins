from os.path import abspath, split, join
import sys

# Create paths
# src_path = split(abspath(__file__))[0]
src_path = '/home/mike/dsi/capstones/predicting_climbing_gym_user_behaviors/src'
prj_path = split(src_path)[0]
ntb_path = join(prj_path, 'notebooks')
data_path = join(split(prj_path)[0], 'data_climb')
img_path = join('/home/mike/dsi/public-repos/weather_and_climbing_gym_checkins/images')