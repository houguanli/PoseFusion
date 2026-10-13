_base_ = '../../../custom_e2e/fine.py'
case_id = 'figurine_d'
pose_id = 3
release_case_config = True
expname = 'figurine_d/pose3'
basedir = './outputs/reconstruction'
WH = [1024, 1024]
expected_image_count = 33
train_all = False
data = dict(datadir='../dataset/figurine_d/pose3', mode=dict(wmask=True))
