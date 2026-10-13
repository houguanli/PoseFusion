_base_ = '../../../custom_e2e/fine.py'
case_id = 'joystick'
pose_id = 2
release_case_config = True
expname = 'joystick/pose2'
basedir = './outputs/reconstruction'
WH = [1024, 1024]
expected_image_count = 32
train_all = False
data = dict(datadir='../dataset/joystick/pose2', mode=dict(wmask=True))
