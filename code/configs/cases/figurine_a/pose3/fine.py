_base_ = '../../../custom_e2e/fine.py'
case_id = 'figurine_a'
pose_id = 3
release_case_config = True
expname = 'figurine_a/pose3'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 45
train_all = False
data = dict(datadir='../dataset/figurine_a/pose3', mode=dict(wmask=True))
