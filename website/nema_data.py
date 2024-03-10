import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

class NEMAData:
    parts = ["li", "ul", "ll", "tt", "tb", "td"]

    m02_min = [[-7.247328,   -50.213535],
               [3.5840948,   -1.4428095],
               [1.4846668,   -55.552578],
               [-34.27443,   -39.35342], 
               [-53.28781,   -30.335218],
               [-77.4173,    -26.710655]]

    m02_max = [[9.046193,    -16.740694],
               [18.87755,    13.447662],
               [25.641842,   -10.108814],
               [-3.6271884,  2.738347],
               [-23.142727,  10.362791],
               [-43.14831,   7.7200465]]

    def __init__(self, ema_data, is_file: bool = True, demean: bool = False, normalize=True) -> None:
        self.ema_data = self.load_data(ema_data, is_file)

        if demean:
            self.demean_center_data()

        self.maya_data = {}

        for part in self.parts:
            self.renormalize_part(part, graph=False)

        if normalize:
            self.normalize_all(normalize_y=False)

            tt_mean = np.mean(self.get_col("tt", 0))
            td_mean = np.mean(self.get_col("td", 0))

            #tt_maya = 8.961
            #td_maya = -9.525
            tt_maya = 7.778
            td_maya = -11.44
            #tt_maya = 6.805
            #td_maya = 1.928

            self.renormalize_to_tongue(tt_mean, td_mean, tt_maya, td_maya)

        for part in self.parts:
            self.maya_data[part] = self.get_formatted_keys(part)

    def load_data(self, data: str, is_file: bool = True) -> dict:
        """
        Loads EMA data from .npy file and separates into each part based on
        information from Peter Wu.

        Args:
            file (str): .npy file to read EMA data from

        Returns:
            ema_data (dict): dictionary with keys for each mouth part and
            values of [x, y] motion capture data
        """
        if is_file:
            loaded_data = np.load(data, allow_pickle=True)
        else:
            loaded_data = data
        ema_data = dict()
        ema_data["li"] = loaded_data[:, 0:2] # lower incisor x, y
        ema_data["ul"] = loaded_data[:, 2:4] # upper lip x, y
        ema_data["ll"] = loaded_data[:, 4:6] # lower lip x, y
        ema_data["tt"] = loaded_data[:, 6:8] # tongue tip x, y
        ema_data["tb"] = loaded_data[:, 8:10] # tongue body x, y
        ema_data["td"] = loaded_data[:, 10:12] # tongue dorsum x, y
        return ema_data

    def get_col(self, part: str, dim: int) -> list:
        """
        Extracts and returns a part's column data.

        Args:
            part (str): mouth part to get data for
            dim (int): 0, 1 corresponding to x, y

        Returns:
            list of coordinates for the chosen part and dim
        """
        return [self.ema_data[part][i][dim] for i in range(0, len(self.ema_data[part]))]

    def get_xy_col(self, part: str) -> Tuple[list, list, list]:
        """
        Extracts and returns a part's x, y column data.

        Args:
            part (str): mouth part to get data for

        Returns:
            tuple of 2 lists of x, y coordinates for the chosen part
        """
        return (self.get_col(part, 0), self.get_col(part, 1))

    def demean_center_data(self) -> None:
        """
        Center EMA data based on total sums of xs and ys since parts are
        not necessarily centered at the origin. Scales to [-1, 1] range of 
        coordinate values.
        """
        sum_xy = [0, 0]
        count_xy = [0, 0]

        max_xy = [0, 0]

        for part in self.parts:
            xy = list(self.get_xy_col(part))

            for i in range(0, 2):
                sum_xy[i] += sum(xy[i])
                count_xy[i] += len(xy[i])

                curr_max = max(abs(min(xy[i])), max(xy[i]))
                if max_xy[i] < curr_max:
                    max_xy[i] = curr_max

        means = [0, 0]
        for i in range(0, 2):
            means[i] = sum_xy[i] / count_xy[i]

        for part in self.parts:
            for i in range(0, len(self.ema_data[part])):
                self.ema_data[part][i] = [
                    (self.ema_data[part][i][0] - means[0]) / (max_xy[0]),
                    (self.ema_data[part][i][1] - means[1]) / (max_xy[1])
                ]

    def get_formatted_keys(self, part: str) -> list:
        """
        Shifts coordinates to match Maya coordinate style.

        Tongue Mocap Data orients coordinates such that x describes back to front,
        y describes right to left, and z describes bottom to top. Maya orients x as
        right to left, y going bottom to top, and z going back to front. As such,
        we reorient (x, y) to (z, y) when using in Maya scripting.

        Additionally, we divide each coordinate by 2 to reduce the overall scale of
        the scene.

        Args:
            part (str): data keyword for body part

        Returns:
            keyframes (list): list of lists containing each keyframe for this part
            as key, x, y in Maya format

        """
        keyframes = []
        xs, ys = self.get_xy_col(part)

        for i in range(0, len(xs)):
            keyframes.append([i, 0, ys[i], xs[i]])

        return keyframes


    def normalize(self, elems: list) -> None:
        """
        Shifts elems into range of [-1, 1] in place.

        Args:
            elems (list): list of coordinates
        """
        orig_min = min(elems)
        orig_max = max(elems)

        for i in range(0, len(elems)):
            elems[i] -= orig_min
            elems[i] /= (orig_max - orig_min)

    def normalize(self, part: str) -> None:
        """
        Shifts elems of given part into range of [-1, 1] in place.

        Args:
            part (str): part from parts
        """
        part_elems = self.ema_data[part]

        x = self.get_col(part, 0)
        y = self.get_col(part, 1)

        for i in range(0, len(part_elems)):
            elem = part_elems[i]
            elem[0] -= min(x)
            elem[0] /= (max(x) - min(x))

            elem[1] -= min(y)
            elem[1] /= (max(y) - min(y))


    def denormalize(self, elems: list, new_min: float, new_max: float) -> None:
        """
        Shifts elems into range of [new_min, new_max] in place.

        Args:
            elems (list): list of coordinates
            new_min (float): new lower bound of elems
            new_max (float): new upper bound of elems
        """
        for i in range(0, len(elems)):
            elems[i] *= (new_max - new_min)
            elems[i] += new_min
    
    def denormalize(self, 
                    part: str, 
                    new_mins: List[float], 
                    new_maxs: List[float]) -> None:
        """
        Shifts elems of given part into range of [new_min, new_max] in place.

        Args:
            part (str): part from parts
            new_mins (tuple): new lower bounds of elems in (x, y) format
            new_maxs (tuple): new upper bounds of elems in (x, y) format
        """
        part_elems = self.ema_data[part]

        for i in range(0, len(part_elems)):
            elem = part_elems[i]
            elem[0] *= (new_maxs[0] - new_mins[0])
            elem[0] += new_mins[0]

            elem[1] *= (new_maxs[1] - new_mins[1])
            elem[1] += new_mins[1]


    def renormalize(self, elems: list, new_min: float, new_max: float) -> None:
        """
        Shifts elems into range of [-1, 1] then [new_min, new_max] in place.

        Args:
            elems (list): list of coordinates
            new_min (float): new lower bound of elems
            new_max (float): new upper bound of elems
        """
        self.normalize(elems)
        self.denormalize(elems, new_min, new_max)

    def renormalize(self, 
                    part: str, 
                    new_mins: List[float], 
                    new_maxs: List[float]) -> None:
        """
        Shifts elems into range of [-1, 1] then [new_min, new_max] in place.

        Args:
            elems (list): list of coordinates
            new_min (float): new lower bound of elems
            new_max (float): new upper bound of elems
        """
        self.normalize(part)
        self.denormalize(part, new_mins, new_maxs)


    def get_part_info(self, part):
        part_elems = self.ema_data[part]

        x = self.get_col(part, 0)
        y = self.get_col(part, 1)

        return part_elems, x, y
    
    def normalize_all(self, normalize_x=True, normalize_y=True):
        first_elems, first_x, first_y = self.get_part_info(self.parts[0])

        overall_min_x = min(first_x)
        overall_min_y = min(first_y)
        overall_max_x = max(first_x)
        overall_max_y = max(first_y)


        for part in self.parts:
            part_elems, x, y = self.get_part_info(part)

            if min(x) < overall_min_x:
                overall_min_x = min(x)
            if min(y) < overall_min_y:
                overall_min_y = min(y)
            if max(x) > overall_max_x:
                overall_max_x = max(x)
            if max(y) > overall_max_y:
                overall_max_y = max(y)

        for part in self.parts:
            part_elems, x, y = self.get_part_info(part)

            for i in range(0, len(part_elems)):
                elem = part_elems[i]

                if normalize_x:
                    elem[0] -= overall_min_x
                    elem[0] /= (overall_max_x - overall_min_x)

                    elem[1] /= (overall_max_x - overall_min_x)

                if normalize_y:
                    elem[1] -= overall_min_y
                    elem[1] /= (overall_max_y - overall_min_y)

    def find_part_index(self, part: str):
        """Returns index in parts of part"""
        for i in range(0, len(self.parts)):
            if self.parts[i] == part:
                return i

    def renormalize_part(self, part: str, graph=True):
        part_index = self.find_part_index(part)

        self.renormalize(part, self.m02_min[part_index], self.m02_max[part_index])

        if graph:
            plt.scatter(self.get_col(part, 0), self.get_col(part, 1), label=part)
            plt.legend()


    def renormalize_to_tongue(self, tt_mean, td_mean, maya_tt, maya_td):
        for part in self.parts:
            part_elems, _, _ = self.get_part_info(part)

            for i in range(0, len(part_elems)):
                elem = part_elems[i]
                self.scale_factor = (maya_tt - maya_td)/(
                    tt_mean - td_mean)
                elem[0] = ((elem[0] - td_mean) 
                           * self.scale_factor) + (maya_td)
                elem[1] *= self.scale_factor

    def offset_li(self):
        li_elems, x, y = self.get_part_info("li")

        first_li_y = li_elems[0][1]
        min_li_y = max(y)
        print(f"first li y: {max(y)}")

        offset = max(y)
        if self.scale_factor:
            print("\n--- scaled offset---")
            print(f"\n--- scale factor ---: {self.scale_factor}")
            offset = self.m02_max[self.find_part_index("li")][1] * self.scale_factor
        print(offset)

        for i in range(0, len(li_elems)):
            elem = li_elems[i]
            #print(f"Original elem[1]: {elem[1]}")
            elem[1] -= max(y)
            #print(f"Offset elem[1]: {elem[1]}\n")

        self.maya_data["li"] = self.get_formatted_keys("li")

    def offset_parts(self, parts):
        for part in parts:
            part_elems, x, y = self.get_part_info(part)

            for i in range(0, len(part_elems)):
                elem = part_elems[i]

                elem[1] -= max(y)
                elem[0] -= max(x)
        
            self.maya_data[part] = self.get_formatted_keys(part)
    
    def get_dim(self, xy_list: list, dim: int) -> list:
        """Get EMA min/maxes list from dim"""
        xy_dim = []
        for i in range(dim, len(xy_list), 2):
            xy_dim.append(xy_list[i])
        return xy_dim

    def shift_keys(frame_data, part, initial_delta):
        keyframes = []
        for keys in frame_data[part]:
            keyframes.append(
                [keys[0] + initial_delta[0], 
                 keys[1] + initial_delta[1], 
                 keys[2] + initial_delta[2], 
                 keys[3] + initial_delta[3]]
            )
        frame_data[part] = keyframes

    
    def shift_all_keys(self, delta):
        for part in self.parts:
            NEMAData.shift_keys(self.maya_data, part, delta)

    def shift_keys_to_tongue(self, part: str, initial_delta: List[int]) -> list:
        NEMAData.shift_keys(self.maya_data, part, initial_delta)

    def save_nema(self, filename):
        np.save(filename, self.ema_data)
    
    def get_json(self):
        json_nema = {part: [] for part in self.parts}
        for part in self.parts:
            part_data = np.array(self.maya_data[part])

            json_nema[part] = part_data[:, 2:4].transpose(1, 0).tolist()
        return json_nema