_base_ = '../../../custom_e2e/fine.py'
case_id = 'guardian'
pose_id = 2
release_case_config = True
expname = 'guardian/pose2'
basedir = './outputs/reconstruction'
WH = [1024, 1024]
expected_image_count = 60
train_all = False
data = dict(datadir='../dataset/guardian/pose2', mode=dict(wmask=True))
