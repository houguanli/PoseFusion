_base_ = '../../../custom_e2e/fine.py'
case_id = 'figurine_e'
pose_id = 2
release_case_config = True
expname = 'figurine_e/pose2'
basedir = './outputs/reconstruction'
WH = [1024, 1024]
expected_image_count = 33
train_all = False
data = dict(datadir='../dataset/figurine_e/pose2', mode=dict(wmask=True))
