self._opendoor_goals = [2,201,202,214,215,10,1001,1002,1014,1015,4,401,402,414,415,12,1201,1202,1214,1215,7,701,702,714,715,15,1501,1502,1514,1515]
self._last_goal = 0

def execute(self, ud):
        def post(id,num):
            requests.post('http://10.255.254.1:18081/v1/ac/door', json=dict(
                            method='open',
                            param=dict(
                                moid=4842000000005000+id,
                                name='C2-4F-冷通道0'+num+'-门禁'
                            )
                        ))
            rospy.loginfo('Open Door: {}'.format(num))

        self.robot.change_state(RobotState.Moving, self._init_state)
        self.robot.reset_as_dock_pose()

        assert self._goal != 0
        if self._goal == self.robot.get_staying_loc() and self._goal > 0:
            return 'standing_by'

        with self.robot.move_to_loc(self._goal, self._open_doors) as moving:
            while not rospy.is_shutdown():
                if self._goal in self._opendoor_goals:
                    if self._opendoor_goals.index(self._goal)<3:
                        if self._last_goal == 1015:
                            post(212, '2-西')
                            post(171, '2-东')
                            post(133, '1-西')
                            post(134, '1-东')
                        else:
                            post(212, '2-西')
                    elif self._opendoor_goals.index(self._goal)<8:
                        post(171, '2-东')
                        post(133, '1-西')
                    elif self._opendoor_goals.index(self._goal)<10:
                        post(134, '1-东')
                    elif self._opendoor_goals.index(self._goal)<13:
                        if self._last_goal == 1215:
                            post(797, '4-西')
                            post(798, '4-东')
                            post(143, '3-西')
                            post(136, '3-东')
                        else:
                            post(797, '4-西')
                    elif self._opendoor_goals.index(self._goal)<18:
                        post(798, '4-东')
                        post(143, '3-西')
                    elif self._opendoor_goals.index(self._goal)<20:
                        post(136, '3-东')
                    elif self._opendoor_goals.index(self._goal)<23:
                        if self._last_goal == 1515:
                            post(139, '6-西')
                            post(138, '6-东')
                            post(161, '5-西')
                            post(160, '6-东')
                        else:
                            post(139, '6-西')
                    elif self._opendoor_goals.index(self._goal)<28:
                        post(138, '6-东')
                        post(161, '5-西')
                    else:
                        post(160, '6-东')
                        
                if self.robot.is_error():
                    self.robot.navigator.cancel_goal()
                    return 'error'
                ctrl = self.robot.get_state_ctrl_cmd()
                if ctrl:
                    if ctrl.type == StateCtrlCmd.StartTakingOver:
                        self.robot.navigator.cancel_goal()
                        return 'taken_over'
                    if ctrl.type == StateCtrlCmd.Stop:
                        self.robot.navigator.cancel_goal()
                        return 'standing_by'

                state = self.robot.navigator.get_state()
                if state == GoalStatus.SUCCEEDED:
                    self._last_goal = self._goal
                    moving.result = Robot.MovingResult.Reached
                    return 'standing_by' if self._goal > 0 else 'charging'
                elif state == GoalStatus.ABORTED:
                    self.robot.set_error(ErrCode.from_navi_error(self.robot.navigator.get_result().err_code))
                    return 'error'

                self.robot.rate.sleep()
