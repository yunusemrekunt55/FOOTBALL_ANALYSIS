from utils import read_video, save_video
from trackers import Tracker
from player_ball_assigner import PlayerBallAssigner

def main():
    
    video_frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('models/best.pt')

     
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    tracks= tracker.get_object_tracks(video_frames, read_from_stub=True,
                                      stub_path='stubs/track_stubs.pkl')

    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    output_video_frames= tracker.draw_annotations(video_frames, tracks)
    
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()