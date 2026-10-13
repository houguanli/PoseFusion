_base_ = '../../../custom_e2e/fine.py'
case_id = 'figurine_b'
pose_id = 2
release_case_config = True
expname = 'figurine_b/pose2'
basedir = './outputs/reconstruction'
WH = [800, 800]
expected_image_count = 40
train_all = False
data = dict(datadir='../dataset/figurine_b/pose2', mode=dict(wmask=True))
