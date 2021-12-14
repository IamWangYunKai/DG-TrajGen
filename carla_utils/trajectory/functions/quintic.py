
import numpy as np

PARAM_NUM = 6
class Quintic(object):
	@staticmethod
	def station_vec(t, t0):
		return np.array([[1, (t-t0), (t-t0)**2, (t-t0)**3, (t-t0)**4, (t-t0)**5]])
	@staticmethod
	def velocity_vec(t, t0):
		return np.array([[0, 1, 2*(t-t0), 3*(t-t0)**2, 4*(t-t0)**3, 5*(t-t0)**4]])
	@staticmethod
	def acceleration_vec(t, t0):
		return np.array([[0, 0, 2, 6*(t-t0), 12*(t-t0)**2, 20*(t-t0)**3]])
	@staticmethod
	def jerk_vec(t, t0):
		return np.array([[0, 0, 0, 6, 24*(t-t0), 60*(t-t0)**2]])

	@staticmethod
	def station_at(rho, t, t0):
		return np.dot(Quintic.station_vec(t, t0), rho)[0][0]
	@staticmethod
	def velocity_at(rho, t, t0):
		return np.dot(Quintic.velocity_vec(t, t0), rho)[0][0]
	@staticmethod
	def acceleration_at(rho, t, t0):
		return np.dot(Quintic.acceleration_vec(t, t0), rho)[0][0]
	@staticmethod
	def jerk_at(rho, t, t0):
		return np.dot(Quintic.jerk_vec(t, t0), rho)[0][0]

	# todo
	@staticmethod
	def station_matrix(t_array, t0):
		TSM = Quintic.station_vec(t_array[0], t0)
		for t in t_array[1:]:
			s_v = Quintic.station_vec(t, t0)
			TSM = np.append(TSM, s_v, axis=0)
		return TSM
	@staticmethod
	def velocity_matrix(t_array, t0):
		TVM = Quintic.velocity_vec(t_array[0], t0)
		for t in t_array[1:]:
			v_v = Quintic.velocity_vec(t, t0)
			TVM = np.append(TVM, v_v, axis=0)
		return TVM
	@staticmethod
	def acceleration_matrix(t_array, t0):
		TAM = Quintic.acceleration_vec(t_array[0], t0)
		for t in t_array[1:]:
			a_v = Quintic.acceleration_vec(t, t0)
			TAM = np.append(TAM, a_v, axis=0)
		return TAM
	@staticmethod
	def jerk_matrix(t_array, t0):
		TJM = Quintic.jerk_vec(t_array[0], t0)
		for t in t_array[1:]:
			j_v = Quintic.jerk_vec(t, t0)
			TJM = np.append(TJM, j_v, axis=0)
		return TJM



class PWQuintic(object):
	@staticmethod
	def get_pw_index(t, pw_t0):
		pw_index = None
		for i in range(len(pw_t0)-1, -1, -1):
			t0p = pw_t0[i]
			if t >= t0p:
				pw_index = i
				break
		return pw_index

	@staticmethod
	def truncate_rho_matrix(pw_num):
		pw_Mr = []
		for i in range(pw_num):
			Mr = np.eye(PARAM_NUM, PARAM_NUM*pw_num, k=PARAM_NUM*i)
			pw_Mr.append(Mr)
		return pw_Mr


	@staticmethod
	def station_at(t, pw_t0, pw_rho):
		pw_num = len(pw_t0)
		pw_Mr = PWQuintic.truncate_rho_matrix(pw_num)
		pw_index = PWQuintic.get_pw_index(t, pw_t0)
		t0, Mr = pw_t0[pw_index], pw_Mr[pw_index]
		rho = np.dot(Mr, pw_rho)
		return Quintic.station_at(rho, t, t0)
