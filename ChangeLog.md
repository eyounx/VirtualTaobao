## [0.0.4]
### Added
- Add RL and SL baselines. Try ```python ./virtualTB/ReinforcementLearning/main.py``` to see more.

## [0.0.3]
### Added
- This changelog file
- Add ./model/LeaveModel.py : predict when a virtual user will leave the vTaobao platform

### Fixed
- Fix the range of env.observation_space
- Now virtual users may not click an item more than once

### Changed
- The user action model now never contains the information of whether user continue browsing any more. Instead we employ a new model which simulates when the user will leave