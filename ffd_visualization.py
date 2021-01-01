from manimlib.imports import *

CON_POINT_RANGE = 7


# B spline basis function
def B_0(u):
    assert 0 <= u <= 1
    return (1. - u) ** 3. / 6.


def B_1(u):
    assert 0 <= u <= 1
    return (3 * u ** 3 - 6 * u ** 2 + 4) / 6


def B_2(u):
    assert 0 <= u <= 1
    return (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6


def B_3(u):
    assert 0 <= u <= 1
    return u ** 3 / 6


def naive_transformation(pos_3d, mesh, delta, B, K):
    pos = pos_3d[0:2]
    pos = (pos - B) / K

    pos_reg = pos / delta
    pos_floor = np.floor(pos_reg)
    uv = pos_reg - pos_floor
    ij = pos_floor - 1.
    B_00 = B_0(uv[0]) * B_0(uv[1])
    B_01 = B_0(uv[0]) * B_1(uv[1])
    B_02 = B_0(uv[0]) * B_2(uv[1])
    B_03 = B_0(uv[0]) * B_3(uv[1])
    B_10 = B_1(uv[0]) * B_0(uv[1])
    B_11 = B_1(uv[0]) * B_1(uv[1])
    B_12 = B_1(uv[0]) * B_2(uv[1])
    B_13 = B_1(uv[0]) * B_3(uv[1])
    B_20 = B_2(uv[0]) * B_0(uv[1])
    B_21 = B_2(uv[0]) * B_1(uv[1])
    B_22 = B_2(uv[0]) * B_2(uv[1])
    B_23 = B_2(uv[0]) * B_3(uv[1])
    B_30 = B_3(uv[0]) * B_0(uv[1])
    B_31 = B_3(uv[0]) * B_1(uv[1])
    B_32 = B_3(uv[0]) * B_2(uv[1])
    B_33 = B_3(uv[0]) * B_3(uv[1])
    B_all = np.array(((B_00, B_01, B_02, B_03),
                      (B_10, B_11, B_12, B_13),
                      (B_20, B_21, B_22, B_23),
                      (B_30, B_31, B_32, B_33)))
    mesh_part = mesh[:, int(ij[0] + 1):int(ij[0] + 1) + 4, int(ij[1] + 1):int(ij[1] + 1) + 4]
    tmp = B_all * mesh_part
    output = np.zeros(3)
    output[0:2] = [(tmp[0, :, :].sum() * K) + B, (tmp[1, :, :].sum() * K) + B]
    return output


class FFDSquare(Scene):
    def __init__(self, mesh, mesh_trans, delta, **scene_kwargs):
        self.mesh = mesh
        self.mesh_trans = mesh_trans
        self.delta = delta
        self.mesh_size = self.mesh.shape[1] - 3
        self.K = CON_POINT_RANGE / (self.mesh_size + 1) / delta
        self.B = CON_POINT_RANGE * (1 / (self.mesh_size + 1) - 0.5)
        self.GEOMETRY_SIZE = self.K * delta * (self.mesh_size - 1) - 0.0001
        self.AXIS_MIN = -0.5 * self.GEOMETRY_SIZE
        self.CONFIG = {
            "x_min": self.AXIS_MIN,
            "x_max": -self.AXIS_MIN,
            "y_min": self.AXIS_MIN,
            "y_max": -self.AXIS_MIN,
            "background_line_style": {
                "stroke_color": "#FFFFFF",
            },
            "x_line_frequency": 0.5 * self.K * delta,
            "y_line_frequency": 0.5 * self.K * delta,
        }
        super().__init__(**scene_kwargs)

    def construct(self):
        control_points = VGroup(*[Dot(point=[self.mesh[0, i, j] * self.K + self.B,
                                             self.mesh[1, i, j] * self.K + self.B,
                                             0])
                                  for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 2)])
        control_points_trans = VGroup(*[Dot(point=[self.mesh_trans[0, i, j] * self.K + self.B,
                                                   self.mesh_trans[1, i, j] * self.K + self.B,
                                                   0])
                                        for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 2)])

        control_lines = VGroup(
            *[Line(np.append(self.mesh[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh[:, i + 1, j] * self.K + self.B, 0))
              for i in range(self.mesh_size + 1) for j in range(self.mesh_size + 2)],
            *[Line(np.append(self.mesh[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh[:, i, j + 1] * self.K + self.B, 0))
              for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 1)]
        )

        control_lines_trans = VGroup(
            *[Line(np.append(self.mesh_trans[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh_trans[:, i + 1, j] * self.K + self.B, 0))
              for i in range(self.mesh_size + 1) for j in range(self.mesh_size + 2)],
            *[Line(np.append(self.mesh_trans[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh_trans[:, i, j + 1] * self.K + self.B, 0))
              for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 1)]
        )

        control_lines.set_color("#70c3ff")
        control_lines_trans.set_color("#70c3ff")

        grid = NumberPlane(**self.CONFIG)
        square_num = 10.
        square_side_length = self.GEOMETRY_SIZE / square_num
        squares = VGroup(
            *[Square(side_length=square_side_length, fill_opacity=1).shift(x * RIGHT + y * UP)
              for x in np.arange(self.AXIS_MIN + 0.5 * square_side_length,
                                 self.AXIS_MIN + self.GEOMETRY_SIZE - 0.4 * square_side_length,
                                 square_side_length)
              for y in np.arange(self.AXIS_MIN + 0.5 * square_side_length,
                                 self.AXIS_MIN + self.GEOMETRY_SIZE - 0.4 * square_side_length,
                                 square_side_length)])

        squares.set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)
        #
        self.add(squares, grid, control_lines, control_points)
        squares.save_state()
        grid.save_state()
        control_points.save_state()
        control_lines.save_state()
        grid.prepare_for_nonlinear_transform()
        self.play(
            Transform(control_points, control_points_trans),
            Transform(control_lines, control_lines_trans),
            ApplyPointwiseFunction(lambda p: naive_transformation(p, self.mesh_trans, self.delta, self.B, self.K),
                                   squares),
            ApplyPointwiseFunction(lambda p: naive_transformation(p, self.mesh_trans, self.delta, self.B, self.K),
                                   grid),
            run_time=1,
        )
        self.play(
            Restore(grid, run_time=1),
            Restore(squares, run_time=1),
            Restore(control_points, run_time=1),
            Restore(control_lines, run_time=1)
        )


class FFDDots(Scene):
    def __init__(self, mesh, mesh_trans, delta, **scene_kwargs):
        self.mesh = mesh
        self.mesh_trans = mesh_trans
        self.delta = delta
        self.mesh_size = self.mesh.shape[1] - 3
        self.K = CON_POINT_RANGE / (self.mesh_size + 1) / delta
        self.B = CON_POINT_RANGE * (1 / (self.mesh_size + 1) - 0.5)
        GEOMETRY_SIZE = self.K * delta * (self.mesh_size - 1) - 0.0001
        self.AXIS_MIN = -0.5 * GEOMETRY_SIZE
        self.CONFIG = {
            "x_min": self.AXIS_MIN,
            "x_max": -self.AXIS_MIN,
            "y_min": self.AXIS_MIN,
            "y_max": -self.AXIS_MIN,
            "background_line_style": {
                "stroke_color": "#FFFFFF",
            },
            "x_line_frequency": 0.5 * self.K * delta,
            "y_line_frequency": 0.5 * self.K * delta,
        }
        super().__init__(**scene_kwargs)

    def construct(self):
        control_lines = VGroup(
            *[Line(np.append(self.mesh[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh[:, i + 1, j] * self.K + self.B, 0))
              for i in range(self.mesh_size + 1) for j in range(self.mesh_size + 2)],
            *[Line(np.append(self.mesh[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh[:, i, j + 1] * self.K + self.B, 0))
              for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 1)]
        )

        control_lines_trans = VGroup(
            *[Line(np.append(self.mesh_trans[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh_trans[:, i + 1, j] * self.K + self.B, 0))
              for i in range(self.mesh_size + 1) for j in range(self.mesh_size + 2)],
            *[Line(np.append(self.mesh_trans[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh_trans[:, i, j + 1] * self.K + self.B, 0))
              for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 1)]
        )

        control_grid = VGroup(control_lines, control_lines_trans)
        control_grid.set_color("#70c3ff")

        grid = NumberPlane(**self.CONFIG)
        dot_radius = 0.5 * 0.25 * self.delta * self.K
        dots = VGroup(
            *[Dot(radius=dot_radius, fill_opacity=1).shift(x * RIGHT + y * UP)
              for x in np.arange(self.AXIS_MIN + 0.25 * self.delta * self.K,
                                 -self.AXIS_MIN - 0.24 * self.delta * self.K,
                                 0.5 * self.delta * self.K)
              for y in np.arange(self.AXIS_MIN + 0.25 * self.delta * self.K,
                                 -self.AXIS_MIN - 0.24 * self.delta * self.K,
                                 0.5 * self.delta * self.K)])

        dots.set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)

        self.add(dots, grid, control_lines)
        dots.save_state()
        grid.save_state()
        control_lines.save_state()
        grid.prepare_for_nonlinear_transform()
        self.play(
            Transform(control_lines, control_lines_trans),
            ApplyPointwiseFunction(lambda p: naive_transformation(p, self.mesh_trans, self.delta, self.B, self.K),
                                   dots),
            ApplyPointwiseFunction(lambda p: naive_transformation(p, self.mesh_trans, self.delta, self.B, self.K),
                                   grid),
            run_time=1,
        )
        self.play(Restore(grid, run_time=1),
                  Restore(dots, run_time=1),
                  Restore(control_lines, run_time=1))


class FFDVectorsWithGrid(Scene):
    def __init__(self, mesh, mesh_trans, delta, **scene_kwargs):
        self.mesh = mesh
        self.mesh_trans = mesh_trans
        self.delta = delta
        self.mesh_size = self.mesh.shape[1] - 3
        self.K = CON_POINT_RANGE / (self.mesh_size + 1) / delta
        self.B = CON_POINT_RANGE * (1 / (self.mesh_size + 1) - 0.5)
        GEOMETRY_SIZE = self.K * delta * (self.mesh_size - 1) - 0.0001
        self.AXIS_MIN = -0.5 * GEOMETRY_SIZE

        self.GEOMETRY_SIZE = self.K * delta * (self.mesh_size - 1) - 0.0001
        self.AXIS_MIN = -0.5 * GEOMETRY_SIZE
        self.CONFIG = {
            "x_min": self.AXIS_MIN,
            "x_max": -self.AXIS_MIN,
            "y_min": self.AXIS_MIN,
            "y_max": -self.AXIS_MIN,
            "background_line_style": {
                "stroke_color": "#FFFFFF",
            },
            "x_line_frequency": 0.5 * self.K * delta,
            "y_line_frequency": 0.5 * self.K * delta,
            "max_stroke_width_to_length_ratio": 10,
        }
        super().__init__(**scene_kwargs)

    def construct(self):
        control_line_width = 1.5
        control_lines = VGroup(
            *[Line(np.append(self.mesh[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh[:, i + 1, j] * self.K + self.B, 0), stroke_width=control_line_width)
              for i in range(self.mesh_size + 1) for j in range(self.mesh_size + 2)],
            *[Line(np.append(self.mesh[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh[:, i, j + 1] * self.K + self.B, 0), stroke_width=control_line_width)
              for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 1)]
        )

        control_lines_trans = VGroup(
            *[Line(np.append(self.mesh_trans[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh_trans[:, i + 1, j] * self.K + self.B, 0), stroke_width=control_line_width)
              for i in range(self.mesh_size + 1) for j in range(self.mesh_size + 2)],
            *[Line(np.append(self.mesh_trans[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh_trans[:, i, j + 1] * self.K + self.B, 0), stroke_width=control_line_width)
              for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 1)]
        )

        control_grid = VGroup(control_lines, control_lines_trans)
        control_grid.set_color("#70c3ff")

        grid = NumberPlane(**self.CONFIG)

        points = [x * RIGHT + y * UP
                  for x in np.arange(self.AXIS_MIN + 0.25 * self.delta * self.K,
                                     -self.AXIS_MIN - 0.24 * self.delta * self.K,
                                     0.5 * self.delta * self.K)
                  for y in np.arange(self.AXIS_MIN + 0.25 * self.delta * self.K,
                                     -self.AXIS_MIN - 0.24 * self.delta * self.K,
                                     0.5 * self.delta * self.K)
                  ]
        vectors = VGroup(*[Vector([0, 0, 0]).shift(point) for point in points])
        scale_factor = 1
        vectors_trans = VGroup(*[Vector(scale_factor *
                                        (naive_transformation(point, self.mesh_trans, self.delta, self.B, self.K) - point),
                                        **self.CONFIG).shift(point)
                                 for point in points])

        vectors_trans.set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE)

        self.add(vectors, grid, control_lines)
        vectors.save_state()
        grid.save_state()
        control_lines.save_state()
        grid.prepare_for_nonlinear_transform()
        self.play(
            Transform(control_lines, control_lines_trans),
            Transform(vectors, vectors_trans),
            ApplyPointwiseFunction(lambda p: naive_transformation(p, self.mesh_trans, self.delta, self.B, self.K),
                                   grid),
            run_time=1,
        )
        self.play(Restore(grid, run_time=1),
                  Restore(vectors, run_time=1),
                  Restore(control_lines, run_time=1))


class FFDVectors(Scene):
    def __init__(self, mesh, mesh_trans, delta, **scene_kwargs):
        self.mesh = mesh
        self.mesh_trans = mesh_trans
        self.delta = delta
        self.mesh_size = self.mesh.shape[1] - 3
        self.K = CON_POINT_RANGE / (self.mesh_size + 1) / delta
        self.B = CON_POINT_RANGE * (1 / (self.mesh_size + 1) - 0.5)
        GEOMETRY_SIZE = self.K * delta * (self.mesh_size - 1) - 0.0001
        self.AXIS_MIN = -0.5 * GEOMETRY_SIZE

        self.GEOMETRY_SIZE = self.K * delta * (self.mesh_size - 1) - 0.0001
        self.AXIS_MIN = -0.5 * GEOMETRY_SIZE
        self.CONFIG = {
            "x_min": self.AXIS_MIN,
            "x_max": -self.AXIS_MIN,
            "y_min": self.AXIS_MIN,
            "y_max": -self.AXIS_MIN,
            "background_line_style": {
                "stroke_color": "#FFFFFF",
            },
            "x_line_frequency": 0.5 * self.K * delta,
            "y_line_frequency": 0.5 * self.K * delta,
            "max_stroke_width_to_length_ratio": 10,
        }
        super().__init__(**scene_kwargs)

    def construct(self):
        control_points = VGroup(*[Dot(point=[self.mesh[0, i, j] * self.K + self.B,
                                             self.mesh[1, i, j] * self.K + self.B,
                                             0], radius=0.05 * self.delta * self.K)
                                  for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 2)])
        control_points_trans = VGroup(*[Dot(point=[self.mesh_trans[0, i, j] * self.K + self.B,
                                                   self.mesh_trans[1, i, j] * self.K + self.B,
                                                   0], radius=0.05 * self.delta * self.K)
                                        for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 2)])
        control_line_width = 1.5
        control_lines = VGroup(
            *[Line(np.append(self.mesh[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh[:, i + 1, j] * self.K + self.B, 0), stroke_width=control_line_width)
              for i in range(self.mesh_size + 1) for j in range(self.mesh_size + 2)],
            *[Line(np.append(self.mesh[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh[:, i, j + 1] * self.K + self.B, 0), stroke_width=control_line_width)
              for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 1)]
        )

        control_lines_trans = VGroup(
            *[Line(np.append(self.mesh_trans[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh_trans[:, i + 1, j] * self.K + self.B, 0), stroke_width=control_line_width)
              for i in range(self.mesh_size + 1) for j in range(self.mesh_size + 2)],
            *[Line(np.append(self.mesh_trans[:, i, j] * self.K + self.B, 0),
                   np.append(self.mesh_trans[:, i, j + 1] * self.K + self.B, 0), stroke_width=control_line_width)
              for i in range(self.mesh_size + 2) for j in range(self.mesh_size + 1)]
        )

        control_grid = VGroup(control_lines, control_lines_trans)
        control_grid.set_color("#70c3ff")

        points = [x * RIGHT + y * UP
                  for x in
                  np.arange(self.AXIS_MIN + 0.25 * self.delta * self.K,
                            -self.AXIS_MIN - 0.24 * self.delta * self.K,
                            0.5 * self.delta * self.K)
                  for y in
                  np.arange(self.AXIS_MIN + 0.25 * self.delta * self.K,
                            -self.AXIS_MIN - 0.24 * self.delta * self.K,
                            0.5 * self.delta * self.K)
                  ]
        vectors = VGroup(*[Vector([0, 0, 0]).shift(point) for point in points])
        scale_factor = 2
        vectors_trans = VGroup(*[Vector(scale_factor *
                                        (naive_transformation(point, self.mesh_trans, self.delta, self.B, self.K) - point),
                                        **self.CONFIG).shift(point)
                                 for point in points])

        vectors_trans.set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE)

        self.add(vectors, control_lines, control_points)
        vectors.save_state()
        control_lines.save_state()
        control_points.save_state()
        self.play(
            Transform(control_lines, control_lines_trans),
            Transform(control_points, control_points_trans),
            Transform(vectors, vectors_trans),
            run_time=1,
        )
        self.play(Restore(vectors, run_time=1),
                  Restore(control_points, run_time=1),
                  Restore(control_lines, run_time=1))
