_base_ = '../../../custom_e2e/coarse.py'
case_id = 'airplane'
pose_id = 3
release_case_config = True
expname = 'airplane/pose3'
basedir = './outputs/reconstruction'
WH = [1024, 1024]
expected_image_count = 32
train_all = True
data = dict(datadir='../dataset/airplane/pose3', mode=dict(wmask=True))
