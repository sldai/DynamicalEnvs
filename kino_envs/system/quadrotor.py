import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/..')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_line_3d(ax, p, p_index, color='b', alpha=1):
    for p_i in p_index:
        ax.plot3D(p[p_i, 0], p[p_i, 1], p[p_i, 2], c=color, alpha=alpha)
        
def centered_box_to_points_3d(center, size):
    half_size = [s/2 for s in size]
    direction, p = [1, -1], []
    for x_d in direction:
        for y_d in direction:
            for z_d in direction:
                p.append([center[di] + [x_d, y_d, z_d][di] * half_size[0] for di in range(3)])
    return p

def rot_frame_3d(state, frame_size=0.25):
    b, c, d, a = state[3:7]
    rot_mat = np.array([[2 * a**2 - 1 + 2 * b**2, 2 * b * c + 2 * a * d, 2 * b * d - 2 * a * c],
                        [2 * b * c - 2 * a * d, 2 * a**2 - 1 + 2 * c**2, 2 * c * d + 2 * a * b],
                        [2 * b * d + 2 * a * c, 2 * c * d - 2 * a * b, 2 * a**2 - 1 + 2 * d**2]])
    quadrotor_frame = np.array([[frame_size, 0, 0],
                                 [0, frame_size, 0],
                                 [-frame_size, 0, 0],
                                 [0, -frame_size, 0]]).T
    quadrotor_frame = rot_mat @ quadrotor_frame + state[:3].reshape(-1, 1)
    return quadrotor_frame

def q_to_points_3d(state):
    quadrotor_frame = rot_frame_3d(state)   
    max_min, direction = [np.max(quadrotor_frame, axis=1), np.min(quadrotor_frame, axis=1)], [1, 0]
    p = []
    for x_d in direction:
        for y_d in direction:
            for z_d in direction:
                p.append([max_min[x_d][0], max_min[y_d][1], max_min[z_d][2]])
    return np.array(p)

def draw_box_3d(ax, p, color='b', alpha=1, surface_color='blue', linewidths=1, edgecolors='k'):
    index_lists = [[[0, 4], [4, 6], [6, 2], [2, 0], [0, 1], [1, 5], [5, 7], [7, 3], [3, 1], [1, 5]],
                  [[4, 5]],
                  [[6, 7]],
                  [[2, 3]]]
    for p_i in index_lists:
        draw_line_3d(ax, np.array(p), p_i, color=color, alpha=alpha)
    edges = [[p[e_i] for e_i in f_i] for f_i in [[0, 1, 5, 4],
                                                 [4, 5, 7, 6],
                                                 [6, 7, 3, 2],
                                                 [2, 0, 1, 3],
                                                 [2, 0, 4, 6],
                                                 [3, 1, 5, 7]]]
    faces = Poly3DCollection(edges, linewidths=linewidths, edgecolors=edgecolors)
    faces.set_facecolor(surface_color)
    faces.set_alpha(0.1)
    ax.add_collection3d(faces)

def draw_quadrotor(ax, state, color='orange'):
    """state format: [x, y, z, quat_x, quat_y, quat_z, w, ...]
    """
    draw_box_3d(ax, q_to_points_3d(state), alpha=0.3, color=color, surface_color=color, linewidths=0., edgecolors=color)

def rot_frame_3d(state, frame_size=0.25):
    b, c, d, a = state[3:7]
    rot_mat = np.array([[2 * a**2 - 1 + 2 * b**2, 2 * b * c + 2 * a * d, 2 * b * d - 2 * a * c],
                        [2 * b * c - 2 * a * d, 2 * a**2 - 1 + 2 * c**2, 2 * c * d + 2 * a * b],
                        [2 * b * d + 2 * a * c, 2 * c * d - 2 * a * b, 2 * a**2 - 1 + 2 * d**2]])
    quadrotor_frame = np.array([[0,0,0] ,
                                [frame_size, 0, 0],
                                [0, frame_size, 0],
                                [-frame_size, 0, 0],
                                [0, -frame_size, 0],
                                [0,0,frame_size]]).T
    quadrotor_frame = rot_mat @ quadrotor_frame + state[:3].reshape(-1, 1)
    return quadrotor_frame

def draw_quadrotor_frame(ax, state, color='orange'):
    frame = rot_frame_3d(state)
    for i in range(1, frame.shape[1]):
        if i == frame.shape[1]-1:
            ax.plot(frame[0, [0, i]], frame[1, [0, i]],
                        frame[2, [0, i]], '-', c=color)
        else:
            ax.plot(frame[0, [0, i]], frame[1, [0, i]],
                        frame[2, [0, i]], '-', c=color)
    
        

class param:
    MIN_C1 = -15. # max trust
    MAX_C1 = -5. # min trust
    MIN_C = -1. # min torque
    MAX_C = 1. # max torque
    MIN_X = -5
    MAX_X = 5
    MIN_V = -1.
    MAX_V = 1.
    MIN_W = -1.
    MAX_W = 1.
    MASS_INV = 1.
    BETA = 1.
    EPS = 2.107342e-08

    radius = 0.25
    goal_radius = 0.5
    width = 1.0
    integration_step = 2e-2
    dt = 0.05

class Quadrotor(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MIN_C1 = -15. # max trust
        self.MAX_C1 = -5. # min trust
        self.MIN_C = -1. # min torque
        self.MAX_C = 1. # max torque
        self.MIN_X = -5
        self.MAX_X = 5
        self.MIN_V = -1.
        self.MAX_V = 1.
        self.MIN_W = -1.
        self.MAX_W = 1.
        self.MASS_INV = 1.
        self.BETA = 1.
        self.EPS = 2.107342e-08
        self.obs_list = []
        self.radius = 0.25
        self.width = 1.0
        self.integration_step = 2e-2

    def enforce_bounds_quaternion(self, qstate):
        # enforce quaternion
        # http://stackoverflow.com/questions/11667783/quaternion-and-normalization/12934750#12934750
        # [x, y, z, w]
        nrmSqr = qstate[0]*qstate[0] + qstate[1]*qstate[1] + qstate[2]*qstate[2] + qstate[3]*qstate[3]
        nrmsq = np.sqrt(nrmSqr) if (np.abs(nrmSqr - 1.0) > 1e-6) else 1.0
        error = np.abs(1.0 - nrmsq)
        if error < self.EPS:
            scale = 2.0 / (1.0 + nrmsq)
            qstate *= scale
        else:
            if nrmsq < 1e-6:
                qstate[:] = 0
                qstate[3] = 1
            else:
                scale = 1.0 / np.sqrt(nrmsq)
                qstate *= scale
        return qstate

    def enforce_bounds_quaternion_vec(self, pose):
        # enforce quaternion
        # http://stackoverflow.com/questions/11667783/quaternion-and-normalization/12934750#12934750
        # [x, y, z, w]
        nrmsq = np.sum(pose ** 2, axis=1)
        ind = np.abs(1.0 - nrmsq) < self.EPS
        pose[ind, :]  *= 2.0 / (1.0 + np.expand_dims(nrmsq[ind], axis=1))
        ind = nrmsq < 1e-6
        pose[ind, 0:3] = 0
        pose[ind, 3] = 1
        pose *= 1.0 / (np.expand_dims(np.sqrt(nrmsq), axis=1) + self.EPS)
        return pose
    
    def _compute_derivatives(self, q, u):
        qdot = np.zeros(q.shape)
        qdot[0:3] = q[7:10]
        qomega = np.zeros(4) #[ x, y, z, w,]
        qomega[0:3] = 0.5 * q[10:13]
        qomega = self.enforce_bounds_quaternion(qomega)
        delta = q[3] * qomega[0] + q[4] * qomega[1] + q[5] * qomega[2]
        qdot[3:7] = qomega - delta * q[3:7]
        qdot[7] = self.MASS_INV * (-2 * u[0] * (q[6] * q[4] + q[3] * q[5]) - self.BETA * q[7])
        qdot[8] = self.MASS_INV * (-2 * u[0] * (q[4] * q[5] - q[6] * q[3]) - self.BETA * q[8])
        qdot[9] = self.MASS_INV * (-u[0] * (q[6] * q[6] - q[3] * q[3] - q[4] * q[4] + q[5] * q[5]) - self.BETA * q[9]) - 9.81
        qdot[10:13] = u[1:4]
        return qdot
    
    def _compute_derivatives_vec(self, q, u):
        qdot = np.zeros(q.shape)
        qdot[:, 0:3] = q[:, 7:10]
        qomega = np.zeros((q.shape[0], 4)) #[ x, y, z, w,]
        qomega[:, 0:3] = 0.5 * q[:, 10:13]
        qomega = self.enforce_bounds_quaternion_vec(qomega)
        delta = q[:, 3] * qomega[:, 0] + q[:, 4] * qomega[:, 1] + q[:, 5] * qomega[:, 2]
        qdot[:, 3:7] = qomega - np.expand_dims(delta, axis=1) * q[:, 3:7]
        qdot[:, 7] = self.MASS_INV * (-2 * u[:, 0] * (q[:, 6] * q[:, 4] + q[:, 3] * q[:, 5]) - self.BETA * q[:, 7])
        qdot[:, 8] = self.MASS_INV * (-2 * u[:, 0] * (q[:, 4] * q[:, 5] - q[:, 6] * q[:, 3]) - self.BETA * q[:, 8])
        qdot[:, 9] = self.MASS_INV * (-u[:, 0] * (q[:, 6] * q[:, 6] - q[:, 3] * q[:, 3] - q[:, 4] * q[:, 4] + q[:, 5] * q[:, 5]) - self.BETA * q[:, 9]) - 9.81
        qdot[:, 10:13] = u[:, 1:4]
        return qdot

    def propagate(self, start_state, control, duration):
        '''
        control (n_sample)
        t is (n_sample)
        # control in [NS, NC=4], t in [NS]
        '''
        steps = int(duration/self.integration_step)
        q = start_state.copy()
        control[0] = np.clip(control[0], self.MIN_C1, self.MAX_C1)
        control[1] = np.clip(control[1], self.MIN_C, self.MAX_C)
        control[2] = np.clip(control[2], self.MIN_C, self.MAX_C)
        control[3] = np.clip(control[3], self.MIN_C, self.MAX_C)
        q[3:7] = self.enforce_bounds_quaternion(q[3:7])
        for t in range(0, steps):
            q += self.integration_step * self._compute_derivatives(q, control)
            q[:3] = np.clip(q[:3], self.MIN_X, self.MAX_X)
            q[7:10] = np.clip(q[7:10], self.MIN_V, self.MAX_V)
            q[10:13] = np.clip(q[10:13], self.MIN_W, self.MAX_W)
            q[3:7] = self.enforce_bounds_quaternion(q[3:7])
        return q


    def propagate_vec(self, start_state, control, t, integration_step, direction=1):
        '''
        control (n_sample)
        t is (n_sample)
        # control in [NS, NC=4], t in [NS]
        '''
        q = start_state
        control[:, 0] = np.clip(control[:, 0], self.MIN_C1, self.MAX_C1)
        control[:, 1] = np.clip(control[:, 1], self.MIN_C, self.MAX_C)
        control[:, 2] = np.clip(control[:, 2], self.MIN_C, self.MAX_C)
        control[:, 3] = np.clip(control[:, 3], self.MIN_C, self.MAX_C)
        q[:, 3:7] = self.enforce_bounds_quaternion_vec(q[:,3:7])
        t_max = np.max(t)
        for t_curr in np.arange(0, t_max + integration_step, integration_step):
            q[ t >= t_curr, :] += direction * integration_step * self._compute_derivatives_vec(q[t >= t_curr, :], control[t >= t_curr, :])
            q[:, 7:10] = np.clip(q[:, 7:10], self.MIN_V, self.MAX_V)
            q[:, 10:13] = np.clip(q[:, 10:13], self.MIN_W, self.MAX_W)
            q[:, 3:7] = self.enforce_bounds_quaternion_vec(q[:, 3:7])
        return q

    def valid_state(self, state):
        for obs in self.obs_list:
            corners = centered_box_to_points_3d(center=obs, size=[self.width]*3)
            obs_min_max = [np.min(corners, axis=0), np.max(corners, axis=0)]
            quadrotor_frame = rot_frame_3d(state, self.radius)   
            quadrotor_min_max = [np.min(quadrotor_frame, axis=1), np.max(quadrotor_frame, axis=1)]
            if quadrotor_min_max[0][0] <= obs_min_max[1][0] and quadrotor_min_max[1][0] >= obs_min_max[0][0] and\
                quadrotor_min_max[0][1] <= obs_min_max[1][1] and quadrotor_min_max[1][1] >= obs_min_max[0][1] and\
                quadrotor_min_max[0][2] <= obs_min_max[1][2] and quadrotor_min_max[1][2] >= obs_min_max[0][2]:
                    return False
        return True


obs_lists = np.array([
    [# 0
        [-1.699173927307129, 2.341871738433838, -3.1702959537506104],
        [-4.387185096740723, 0.6952052712440491, -0.733799934387207],
        [0.72575843334198, 3.395219326019287, -1.2938487529754639],
        [-1.335496187210083, -3.7240636348724365, -4.096003532409668],
        [-3.3221275806427, -3.001746416091919, -2.434605121612549],
        [-0.8298090100288391, 0.5715386271476746, 1.917634129524231],
        [1.3729902505874634, -3.43143630027771, 1.6705354452133179],
        [-0.4881708025932312, 1.9405572414398193, -4.258523464202881],
        [4.201957702636719, 0.22444848716259003, -3.4626381397247314],
        [-3.9318063259124756, -1.3446141481399536, 4.3727312088012695]],
    
    [# 1
        [-0.7672719359397888, -2.9730443954467773, -2.309258460998535],
        [-4.972418308258057, 0.5445275902748108, 1.4016530513763428],
        [1.8297779560089111, -3.569312572479248, 2.280195951461792],
        [0.9630254507064819, 0.2347477525472641, 0.32103654742240906],
        [-1.1161229610443115, -1.8127678632736206, 2.351191282272339],
        [2.745652198791504, 3.839008331298828, 1.6986621618270874],
        [0.4301983714103699, -0.7100529074668884, -3.692326068878174],
        [-3.5266897678375244, -2.537790536880493, 4.041909694671631],
        [4.649423599243164, -1.4894638061523438, -2.3240420818328857],
        [-0.7666326761245728, -1.4784198999404907, 3.9963488578796387]],
    
    [# 2
        [0.8249884843826294, 1.5508390665054321, -3.3601412773132324],
        [-3.090625524520874, -0.8205899596214294, 2.8024654388427734],
        [-1.3554250001907349, -2.350876808166504, -2.275745391845703],
        [-2.0441842079162598, -3.2917003631591797, 1.6558321714401245],
        [1.4645555019378662, -2.3792953491210938, 2.450481414794922],
        [-3.2720439434051514, 2.7424821853637695, 2.255465507507324],
        [0.30334556102752686, 0.7566931247711182, 3.1123087406158447],
        [1.3233407735824585, 3.8840198516845703, 3.196009874343872],
        [-1.3217675685882568, 2.620227813720703, -0.8136425018310547],
        [3.884197235107422, 0.5988503694534302, -0.32626771926879883]],
    [# 3
        [-2.3259687423706055, 0.022344231605529785, 2.3487050533294678],
        [0.8060741424560547, -1.6393802165985107, 0.522911012172699],
        [2.2057902812957764, 2.4950172901153564, 3.6335458755493164],
        [3.7807250022888184, 2.903716564178467, 0.19939565658569336],
        [2.5789241790771484, -3.4462666511535645, 2.630343437194824],
        [-3.833881139755249, 3.634984016418457, 3.0537726879119873],
        [-2.890171527862549, -1.8739043474197388, -1.516227126121521],
        [-1.519995093345642, -3.904817581176758, 4.397702693939209],
        [2.9323105812072754, -1.0947999954223633, -1.3545935153961182],
        [2.7486283779144287, 1.5394071340560913, -2.889136791229248]],
    
    [# 4
        [0.9078376293182373, -2.8725967407226562, 2.177410125732422],
        [0.8003172874450684, 1.849447250366211, 2.245795249938965],
        [-4.7661638259887695, 1.155346393585205, 1.8780364990234375],
        [-3.3690831661224365, 4.1074371337890625, 0.9700890779495239],
        [-1.5302644968032837, 3.670963764190674, -2.402881145477295],
        [-2.2121617794036865, -0.1729590892791748, 4.331843376159668],
        [1.5569299459457397, 4.572744369506836, -0.4823128879070282],
        [3.592019557952881, -1.4648520946502686, -1.5855196714401245],
        [-1.922871708869934, -2.9947144985198975, 0.2990763783454895],
        [4.305208683013916, 2.780130624771118, 0.3179897665977478]],
    
    [# 5
        [3.932040214538574, 2.98964786529541, -4.274775981903076],
        [-4.677023410797119, 3.0797226428985596, 2.49385666847229],
        [0.8901069760322571, -3.3287038803100586, 2.3244853019714355],
        [4.596564769744873, 0.03568652272224426, 0.30531755089759827],
        [1.9519561529159546, -1.8215734958648682, 4.4674248695373535],
        [-0.7722808122634888, 1.8221392631530762, -2.69989013671875],
        [-3.3105950355529785, 1.8422129154205322, -0.8147218227386475],
        [-3.7825264930725098, -1.6278599500656128, 1.0201404094696045],
        [0.4003630578517914, 0.8936196565628052, 0.5970503687858582],
        [-2.883539915084839, -3.8565709590911865, -3.629195213317871]],
    
    [# 6
        [4.21366024017334, -2.6164419651031494, -2.6665797233581543],
        [2.0054759979248047, -5.553534507751465, 2.891672134399414],
        [4.250123977661133, 0.07943809032440186, -3.6494979858398438],
        [0.06250333786010742, 0.16598296165466309, 2.42751407623291],
        [2.08906888961792, -2.361638307571411, -0.16324998438358307],
        [-4.228981018066406, -1.289161205291748, -1.7596478462219238],
        [4.704349517822266, 2.6830177307128906, -1.0154402256011963],
        [-0.03496050834655762, 3.109302043914795, -1.6209912300109863],
        [-4.98598575592041, -4.265467643737793, -0.8945780992507935],
        [-2.251688003540039, 1.4826812744140625, 0.17760059237480164]],
    
    [# 7
        [0.4736948609352112, 5.599517345428467, 0.7380019426345825],
        [-4.742375373840332, 4.7725300788879395, 0.3376665711402893],
        [1.7334680557250977, -1.79729163646698, 2.779433012008667],
        [-2.5241730213165283, -1.7462395429611206, 3.885915756225586],
        [-2.859957218170166, -2.6708836555480957, -4.488961696624756],
        [1.4670977592468262, -4.486936569213867, -2.076280355453491],
        [4.143749237060547, 3.00651478767395, -4.547025680541992],
        [-1.7607990503311157, 0.911713719367981, -3.5916507244110107],
        [1.358733892440796, 1.1290783882141113, -3.780313730239868],
        [1.529350996017456, 1.9267163276672363, -0.16129231452941895]],
    [# 8
        [2.4348959922790527, -2.6967926025390625, 2.6798667907714844],
        [4.645354747772217, 1.1833171844482422, -3.8463265895843506],
        [-4.716761112213135, 0.26634883880615234, 0.9945513010025024],
        [-4.1186113357543945, 0.3095625638961792, -2.732757806777954],
        [-0.7085107564926147, -0.8654947280883789, -1.2701036930084229],
        [3.283475875854492, 5.04448127746582, 1.6169414520263672],
        [0.24358093738555908, 3.3553342819213867, 2.706495523452759],
        [-3.5179572105407715, -2.3180525302886963, 3.667930841445923],
        [-1.6755118370056152, 2.8872148990631104, 4.012903213500977],
        [2.3854787349700928, 1.3170700073242188, 0.3831551969051361]],
    [# 
        [-4.727511882781982, -1.4811898469924927, 4.50010871887207],
        [0.03359222412109375, 2.55497407913208, 0.941753625869751],
        [4.038962364196777, 1.195197582244873, -0.4285720884799957],
        [0.1730407476425171, -0.48836225271224976, 3.5176093578338623],
        [4.47713565826416, -2.5211944580078125, 0.45455580949783325],
        [-3.9160780906677246, 4.030087471008301, 3.05521559715271],
        [2.4322707653045654, -2.400113582611084, -4.049161434173584],
        [3.363713502883911, -1.8379610776901245, 3.9432785511016846],
        [4.921741008758545, 0.37511542439460754, -3.370579719543457],
        [-3.8232808113098145, 2.6178812980651855, -2.225541830062866]],
    ])