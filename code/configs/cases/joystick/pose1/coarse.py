_base_ = '../../../custom_e2e/coarse.py'
case_id = 'joystick'
pose_id = 1
release_case_config = True
expname = 'joystick/pose1'
basedir = './outputs/reconstruction'
WH = [1024, 1024]
expected_image_count = 32
train_all = True
data = dict(datadir='../dataset/joystick/pose1', mode=dict(wmask=True))
