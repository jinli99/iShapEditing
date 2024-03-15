import copy
import os
import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering
import numpy as np
from drag_utils_layer import DragStuff
import torch as th
import threading
from skimage import io


# global points coordinate
coord0 = np.array([[[0.37670793358295884, 0.1450980392156862, 0.06666666666666665], [0.36984926, 0.25788742, 0.05303467]],
                   [[0.03529411764705892, 0.15294117647058814, 0.3767331437165149], [0.027450980392156765, 0.08235294117647052, 0.3784762524549461]],
                   [[0.019607843137254832, -0.006835167338173909, 0.37254901960784315], [0.015053239, -0.09474977, 0.3754375]],
                   [[0.37465164084608205, 0.15294117647058814, 0.050980392156862786], [0.47886518, 0.10046293, 0.022452282]]])

coord1 = np.array([[[-0.5746937322446048, 0.9607843137254901, -0.019607843137254943],[-0.4321638, 0.99389905, -0.02291296]],
                   [[-0.5294117647058824, 0.45882352941176463, 0.685554365270312],[-0.4269995167413214, 0.46666666666666656, 0.7019607843137254]],
                   [[0.6433290319036518, 0.13725490196078427, 0.6784313725490196],[0.6515949383900947, 0.13725490196078427, 0.5764705882352941]],
                   [[-0.019607843137254943, 0.7295395201275006, 0.7019607843137254],[-0.0117647058823529, 0.6705882352941177, 0.7080237083962153]]])

coord2 = np.array([[[-0.6, 0.9396358916769973, -0.050980392156862786],[-0.50043005, 0.96048087, -0.07294896]],
                  [[-0.584313725490196, -0.9372549019607843, 0.6088443311411176],[-0.44125116, -0.92913884, 0.6377995]]])

coord3 = np.array([[[0.7855891090110902, -0.48235294117647054, 0.03529411764705892],[0.7849249253143464, -0.388235294117647, 0.03529411764705892]],
                  [[0.7822867418371218, 0.3803921568627451, 0.7254901960784315],[0.7844279949547688, 0.37254901960784315, 0.8196078431372549]],
                   [[0.780392156862745, 0.1055819936261968, 0.050980392156862786],[0.7833097481120848, 0.019607843137254832, 0.04313725490196085]]])

coord4 = np.array([[[-0.5502471024611093, 0.9529411764705882, -0.019607843137254943],[-0.40328434, 0.98134863, -0.022581317]],
                  [[0.6235294117647059, -0.5618531629667611, -0.0117647058823529],[0.610758, -0.6575152, -0.0067606196]]])

coord5 = np.array([[[0.3176470588235294, -0.45882352941176474, 0.18286770711388556], [0.31836328453498486, -0.45882352941176474, 0.2549019607843137]]])

coord6 = np.array([[[0.03529411764705892, 0.16355255639967758, 0.3019607843137255],[0.035023257, 0.06932815, 0.30845296]]])

coord7 = np.array([[[-0.7803921568627451, -0.9215686274509804, 0.827264931834111],[-0.61606824, -0.9226783, 0.82864094]],
                   [[0.09019607843137245, -0.1686274509803921, 0.8136965670078824],[0.088657185, -0.26755452, 0.8193318]],
                   [[-0.584313725490196, 0.607843137254902, 0.8229115737542192],[-0.6627450980392157, 0.5897420655441326, 0.8274509803921568]]])

coord8 = np.array([[[-0.6293484629480813, 0.9137254901960785, -0.06666666666666665],[-0.47340366, 0.93652105, -0.07335004]]])

coord9 = np.array([[[0.08235294117647052, -0.10588235294117643, 0.47323799932816546],[0.07425436, -0.18789564, 0.4774843]],
                  [[-0.4549887380255814, 0.7960784313725491, 0.0117647058823529],[-0.42148963, 0.6957268, 0.018102137]],
                   [[0.5294117647058822, -0.008207950429874877, -0.019607843137254943],[0.4509803921568627, -0.00010042781378427623, -0.019607843137254943]]])
total_coord = [coord0, coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8, coord9]


class App:

    def __init__(self):
        self._id = 0
        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window("DragShape", 1600, 1200)
        w = self.window
        em = w.theme.font_size

        # Panel Window
        self._panel = gui.Vert(3*em, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.254 * em))

        # Model
        self._model_panel = gui.CollapsableVert('Model & Latent', em)
        self._model_path_panel = gui.Horiz()
        self._model_label = gui.Label('Model')
        self._model_path = gui.Combobox()
        self._model_path.add_item('None')
        self._model_path.add_item('./models/chairs')
        self._model_path.add_item('./models/cars')
        self._model_path.add_item('./models/planes')
        self._model_path_panel.add_child(self._model_label)
        self._model_path_panel.add_child(self._model_path)
        self._latent_panel = gui.Horiz()
        self._latent_label = gui.Label('Latent')
        self._latent_view = gui.NumberEdit(gui.NumberEdit.INT)
        self._latent_view.int_value = 0
        self._latent_panel.add_child(self._latent_label)
        self._latent_panel.add_child(self._latent_view)
        self._reset_create_panel = gui.Horiz()
        self._create_mesh = gui.Button('Create Mesh')
        self._reset_mesh = gui.Button('Reset Mesh')
        self._create_mesh.vertical_padding_em = 0
        self._create_mesh.background_color = gui.Color(0.25, 0.41, 1.0)
        self._reset_mesh.vertical_padding_em = 0
        self._reset_mesh.background_color = gui.Color(0.25, 0.41, 1.0)
        self._reset_create_panel.add_child(self._create_mesh)
        self._reset_create_panel.add_fixed(0.2 * em)
        self._reset_create_panel.add_child(self._reset_mesh)
        self._load_inversion_panel = gui.Horiz()
        self._load_mesh = gui.Button(' Load Mesh  ')
        self._load_mesh.vertical_padding_em = 0
        self._load_mesh.background_color = gui.Color(0.25, 0.41, 1.0)
        self._inversion_mesh = gui.Button('   Inversion  ')
        self._inversion_mesh.vertical_padding_em = 0
        self._inversion_mesh.background_color = gui.Color(0.25, 0.41, 1.0)
        self._load_inversion_panel.add_child(self._load_mesh)
        self._load_inversion_panel.add_fixed(0.2 * em)
        self._load_inversion_panel.add_child(self._inversion_mesh)
        self._model_panel.add_child(self._model_path_panel)
        self._model_panel.add_child(self._latent_panel)
        self._model_panel.add_child(self._reset_create_panel)
        self._model_panel.add_child(self._load_inversion_panel)
        self._print_label = gui.Label('Select a Model')
        self._print_label.text_color = gui.Color(1., 0.65, 0)
        self._model_panel.add_child(self._print_label)

        # Drag
        self._drag_panel = gui.CollapsableVert('Drag', em)
        self._train_panel = gui.Horiz()
        self._train_label = gui.Label('Train')
        self._train_start_btn = gui.Button('Start')
        self._train_start_btn.vertical_padding_em = 0
        self._train_start_btn.background_color = gui.Color(0.25, 0.41, 1.0)
        self._train_stop_btn = gui.Button('Stop')
        self._train_stop_btn.vertical_padding_em = 0
        self._train_stop_btn.background_color = gui.Color(0.25, 0.41, 1.0)
        self._train_panel.add_child(self._train_label)
        self._train_panel.add_fixed(0.7 * em)
        self._train_panel.add_child(self._train_start_btn)
        self._train_panel.add_fixed(0.5 * em)
        self._train_panel.add_child(self._train_stop_btn)
        self._points_panel = gui.Horiz()
        self._points_label = gui.Label('Points')
        self._points_undo_btn = gui.Button('Undo')
        self._points_undo_btn.vertical_padding_em = 0
        self._points_undo_btn.background_color = gui.Color(0.25, 0.41, 1.0)
        self._points_clear_btn = gui.Button('Clear')
        self._points_clear_btn.vertical_padding_em = 0
        self._points_clear_btn.background_color = gui.Color(0.25, 0.41, 1.0)
        self._points_panel.add_child(self._points_label)
        self._points_panel.add_fixed(0.1 * em)
        self._points_panel.add_child(self._points_undo_btn)
        self._points_panel.add_fixed(0.5 * em)
        self._points_panel.add_child(self._points_clear_btn)
        self._drag_panel.add_child(self._train_panel)
        self._drag_panel.add_child(self._points_panel)
        self._params_panel = gui.Horiz()
        self._grads_label = gui.Label('Scale')
        self._grads_scale_edit = gui.TextEdit()
        self._grads_scale_edit.text_value = "1200"
        self._lambda_label = gui.Label('Lambda')
        self._lambda_edit = gui.TextEdit()
        self._lambda_edit.text_value = '0.4'
        self._params_panel.add_child(self._grads_label)
        self._params_panel.add_child(self._grads_scale_edit)
        self._params_panel.add_fixed(0.1*em)
        self._params_panel.add_child(self._lambda_label)
        self._params_panel.add_child(self._lambda_edit)
        self._drag_panel.add_child(self._params_panel)
        self._progress_panel = gui.Horiz()
        self._progres_bar = gui.ProgressBar()
        self._progres_bar.value = 0.0
        self._progres_label = gui.Label("Progress 0%  ")
        self._progress_panel.add_child(self._progres_label)
        self._progress_panel.add_child(self._progres_bar)
        self._drag_panel.add_child(self._progress_panel)

        self._drag_fix_panel = gui.Horiz()
        self._fix_pnt = gui.NumberEdit(gui.NumberEdit.INT)
        self._fix_pnt.int_value = 0
        self._fix_pnt_btn = gui.Button('draw')
        self._fix_pnt_btn.vertical_padding_em = 0
        # self._drag_fix_panel.add_child(self._fix_pnt)
        # self._drag_fix_panel.add_child(self._fix_pnt_btn)
        # self._drag_panel.add_child(self._drag_fix_panel)

        # Capture
        self._capture_panel = gui.CollapsableVert('Capture', 0)
        self._save_panel = gui.Horiz()
        self._save_mesh = gui.Button('  Save Mesh  ')
        self._save_mesh.vertical_padding_em = 0
        self._save_mesh.background_color = gui.Color(0.25, 0.41, 1.0)
        self._save_pic = gui.Button('  Save Pic  ')
        self._save_pic.vertical_padding_em = 0
        self._save_pic.background_color = gui.Color(0.25, 0.41, 1.0)
        self._save_panel.add_child(self._save_mesh)
        self._save_panel.add_fixed(0.2 * em)
        self._save_panel.add_child(self._save_pic)
        self._capture_panel.add_child(self._save_panel)

        self._panel.add_child(self._model_panel)
        self._panel.add_child(self._drag_panel)
        self._panel.add_child(self._capture_panel)

        # Geometry Window
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background([1, 1, 1, 1])
        self._scene.scene.scene.set_sun_light(
            [0, -1, 0],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self._scene.scene.scene.enable_sun_light(True)
        self.drag_stuff = DragStuff()

        # Callback
        self._scene.set_on_mouse(self._mouse_event)
        self._points_undo_btn.set_on_clicked(self._points_undo_callback)
        self._points_clear_btn.set_on_clicked(self._points_clear_callback)
        self._model_path.set_on_selection_changed(self._model_select_callback)
        self._create_mesh.set_on_clicked(self._create_mesh_callback)
        self._reset_mesh.set_on_clicked(self._reset_mesh_callback)
        self._save_mesh.set_on_clicked(self._save_mesh_callback)
        self._save_pic.set_on_clicked(self._save_pic_callback)
        self._load_mesh.set_on_clicked(self._load_mesh_callback)
        self._inversion_mesh.set_on_clicked(self._inversion_callback)
        self._train_start_btn.set_on_clicked(self._train_start_callback)
        self._train_stop_btn.set_on_clicked(self._train_stop_callback)
        self._fix_pnt_btn.set_on_clicked(self._fix_pnt_callback)

        # parameters
        self.source_pnt = []
        self.target_pnt = []
        self.draw_source_flag = True
        self.training_thread = None
        self.source_depth = None
        self.mesh = None
        self.mesh_kdtree = None
        self._print_label_text = ''
        self._progress_value = 0.
        self.abs_dir = os.getcwd()
        self.update_flag = True

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._panel)

    def _on_layout(self, layout_context):

        # Position
        r = self.window.content_rect
        panel_width = self._panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).width*3.3
        panel_height = r.height
        self._panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, panel_height)
        self._scene.frame = gui.Rect(r.x, r.y, r.width-panel_width, r.height)

    def _fix_pnt_callback(self):
        self._points_clear_callback()
        num = total_coord[self._latent_view.int_value].shape[0]
        pnt = total_coord[self._latent_view.int_value][self._fix_pnt.int_value % num]  # 2*3
        self.source_pnt.append(pnt[0])
        self.target_pnt.append(pnt[1])
        self.draw_point(pnt[0], rgb=(1, 0, 0), name="start1")
        self.draw_point(pnt[1], rgb=(0, 0, 1), name="end1" + str(len(self.target_pnt)))
        self.draw_arrow(self.source_pnt[-1], self.target_pnt[-1], name='line1')

    def _print_label_text_fun(self):
        self._print_label.text = self._print_label_text

    def _set_progress_value_fun(self):
        self._progres_bar.value = self._progress_value
        self._progres_label.text = "Progress {value}%".format(value=int(self._progress_value*100))

    def _points_undo_callback(self):
        if len(self.source_pnt) == 0:
            return
        if self.draw_source_flag:
            name_list = ['end'+str(len(self.target_pnt)), 'line'+str(len(self.target_pnt))]
            self.remove_geometry_name(name_list)
            self.target_pnt.pop()
        else:
            self.remove_geometry_name(['start'+str(len(self.source_pnt))])
            self.source_pnt.pop()
        self.draw_source_flag = not self.draw_source_flag

    def _points_clear_callback(self):
        name_list = (['start%d' % i for i in range(1, len(self.source_pnt)+1)] +
                     ['end%d' % i for i in range(1, len(self.target_pnt)+1)] +
                     ['line%d' % i for i in range(1, len(self.target_pnt)+1)])
        self.remove_geometry_name(name_list)
        self.source_pnt.clear()
        self.target_pnt.clear()
        self.draw_source_flag = True

    def _model_select_callback(self, new_val, new_idx):
        self.clear_all()
        self.drag_stuff.clear_params()

        if new_idx == 0:
            self._print_label_text = 'Select a Model'
            gui.Application.instance.post_to_main_thread(self.window, self._print_label_text_fun)
        else:
            self._print_label_text = 'Loading Model...'
            gui.Application.instance.post_to_main_thread(self.window, self._print_label_text_fun)

            def update_model_params():
                t = threading.Thread(target=self.drag_stuff.update_model_params, args=(new_val,))
                t.start()
                t.join()
                self._print_label_text = 'Loading Model Done!'
                gui.Application.instance.post_to_main_thread(self.window, self._print_label_text_fun)
            threading.Thread(target=update_model_params).start()

    def _create_mesh_callback(self):
        if self._model_path.selected_index:
            self.clear_all()
            self.drag_stuff.clear_params()
            self._print_label_text = 'Create Mesh...'
            gui.Application.instance.post_to_main_thread(self.window, self._print_label_text_fun)

            def create_mesh():
                # DDIM latent code
                # if self._model_path.selected_index == 1:
                #     noise = th.tensor(np.load('./datas/noise_chairs.npy'))
                #     # noise = th.tensor(np.load('noise.npy'))
                #     # noise = th.tensor(np.load('./datas/noise_gt_total.npy'))
                # elif self._model_path.selected_index == 2:
                #     noise = th.tensor(np.load('./datas/noise_cars.npy'))
                # else:
                #     noise = th.tensor(np.load('./datas/noise_planes.npy'))
                # t = threading.Thread(target=self.drag_stuff.update_latent_params,
                #                      args=(noise[[self._latent_view.int_value % noise.shape[0]]], ))

                np.random.seed(self._latent_view.int_value)
                t = threading.Thread(target=self.drag_stuff.update_latent_params,
                                     args=(np.random.randn(
                                         1, 96, self.drag_stuff.args.image_size, self.drag_stuff.args.image_size),))
                t.start()
                t.join()
                self.update_mesh(self.drag_stuff.mesh)
                self._print_label_text = 'Create Mesh Done!'
                gui.Application.instance.post_to_main_thread(self.window, self._print_label_text_fun)

            threading.Thread(target=create_mesh).start()

    def _reset_mesh_callback(self):
        if self.drag_stuff.mesh0 is None:
            mesh = copy.deepcopy(self.mesh)
            self.clear_all()
            self.update_mesh(mesh)
        else:
            self.clear_all()
            self.update_mesh(self.drag_stuff.mesh0)
            self.drag_stuff.reset_params()

    def _save_mesh_callback(self):
        if self._scene.scene.has_geometry('mesh'):
            assert self.mesh is not None
            file_picker = gui.FileDialog(gui.FileDialog.SAVE, "Save mesh...", self.window.theme)
            file_picker.add_filter('.obj', 'obj')
            file_picker.add_filter('.ply', 'ply')
            file_picker.add_filter('.off', 'off')
            file_picker.add_filter('.stl', 'stl')
            file_picker.set_path('./datas')
            file_picker.set_on_cancel(self._on_cancel)
            file_picker.set_on_done(self._save_mesh_done)

            # show the dialog
            self.window.show_dialog(file_picker)

    def _save_pic_callback(self):
        if self._scene.scene.has_geometry('mesh'):
            assert self.mesh is not None
            file_picker = gui.FileDialog(gui.FileDialog.SAVE, "Save Pic...", self.window.theme)
            file_picker.add_filter('.png', 'png')
            file_picker.add_filter('.jpg', 'jpg')
            file_picker.set_path('./datas')
            file_picker.set_on_cancel(self._on_cancel)
            file_picker.set_on_done(self._save_pic_done)

            # show the dialog
            self.window.show_dialog(file_picker)

    def _save_pic_done(self, filename):

        # case1: save as png with fix resolution
        img_arr = []

        def show_img(img):
            # o3d.io.write_image(filename, img)
            img_arr.append(np.asarray(img))
        self._scene.scene.scene.render_to_image(show_img)

        def depth_callback(depth_image):
            depth_image_arr = np.asarray(depth_image)
            img_arr[0][depth_image_arr == 1.] = np.array([255, 255, 255], dtype=np.uint8)
            io.imsave(filename, img_arr[0])

        self._scene.scene.scene.render_to_depth_image(depth_callback)

        # case2: set high resolution 2048*2048
        # img = gui.Application.instance.render_to_image(self._scene.scene, 2048, 2048)
        # o3d.io.write_image(filename, img)

        self.window.close_dialog()
        os.chdir(self.abs_dir)

    def _on_cancel(self):
        self.window.close_dialog()

    def _save_mesh_done(self, filename):
        if self.mesh.has_vertex_normals() or self.mesh.has_triangle_normals:
            new_mesh = copy.deepcopy(self.mesh)
            new_mesh.triangle_normals = o3d.utility.Vector3dVector([])
            new_mesh.vertex_normals = o3d.utility.Vector3dVector([])
        else:
            new_mesh = self.mesh
        o3d.io.write_triangle_mesh(filename, new_mesh)
        # o3d.io.write_triangle_mesh(filename, self.mesh, write_vertex_colors=False, write_vertex_normals=False)
        self.window.close_dialog()
        os.chdir(self.abs_dir)

    def _load_mesh_callback(self):
        file_picker = gui.FileDialog(gui.FileDialog.OPEN, "Select mesh...", self.window.theme)
        file_picker.add_filter('.obj', 'obj')
        file_picker.add_filter('.ply', 'ply')
        file_picker.add_filter('.off', 'off')
        file_picker.add_filter('.stl', 'stl')
        file_picker.set_path('./datas')
        file_picker.set_on_cancel(self._on_cancel)
        file_picker.set_on_done(self._load_mesh_done)

        # show the dialog
        self.window.show_dialog(file_picker)

    def _load_mesh_done(self, filename):
        if filename.endswith((".obj", ".ply", ".off", ".stl")):
            print("load", filename)
            # scale and translate the mesh into the [-1, 1]*[-1, 1]*[-1, 1]
            mesh = o3d.io.read_triangle_mesh(filename).filter_smooth_simple(number_of_iterations=20)
            max_bound, min_bound = mesh.get_max_bound(), mesh.get_min_bound()
            axis_extent = max_bound - min_bound
            if np.any(min_bound > 1) or np.any(min_bound < -1) or np.any(max_bound > 1) or np.any(max_bound < -1):
                mesh.translate(-mesh.get_center())
                if axis_extent.max() > 2:
                    mesh.scale(2. / (axis_extent.max() + 1e-2), center=np.array([0., 0, 0]))
            self.clear_all()
            self.drag_stuff.clear_params()
            self.update_mesh(mesh, update_camera=self.update_flag)
            self.window.close_dialog()
            os.chdir(self.abs_dir)
            self.update_flag = False
        else:
            print("Unsupported file type")

    def _inversion_callback(self):
        self._print_label_text = 'IDDPM Inversion...'
        gui.Application.instance.post_to_main_thread(self.window, self._print_label_text_fun)

        def mesh_inversion():
            t = threading.Thread(target=self.drag_stuff.train_triplane, args=(self.mesh, None, False))
            t.start()
            t.join()
            self._print_label_text = 'Inversion Done!'
            gui.Application.instance.post_to_main_thread(self.window, self._print_label_text_fun)

        threading.Thread(target=mesh_inversion).start()

    def _train_start_callback(self):

        if self.mesh is not None and self.drag_stuff.mesh is None:
            self._print_label_text = 'Click "Inversion" button before editing!'
            gui.Application.instance.post_to_main_thread(self.window, self._print_label_text_fun)
            return

        self._progress_value = 0.
        gui.Application.instance.post_to_main_thread(self.window, self._set_progress_value_fun)

        def training():
            # guidance editing
            for progress_value in self.drag_stuff.training(
                    scale=int(self._grads_scale_edit.text_value), cof=float(self._lambda_edit.text_value)):
                self._progress_value = progress_value
                gui.Application.instance.post_to_main_thread(self.window, self._set_progress_value_fun)
            self.update_mesh(self.drag_stuff.mesh, update_camera=False)
            print("Complete!")

        self.training_thread = threading.Thread(target=training)
        self.training_thread.start()

    def _train_stop_callback(self):
        if self.training_thread is not None and self.training_thread.is_alive():
            self.drag_stuff.train_flag = False
            self.training_thread.join()

    def _mouse_event(self, event):
        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT) and
                event.is_modifier_down(gui.KeyModifier.CTRL)):

            def depth_callback(depth_image):
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y

                depth = np.asarray(depth_image)[y, x]
                if depth == 1.0:
                    if self.draw_source_flag:
                        return
                    else:
                        picked_point = np.asarray(self._scene.scene.camera.unproject(
                            x, y, self.source_depth, self._scene.frame.width, self._scene.frame.height))
                else:
                    world_coord = self._scene.scene.camera.unproject(
                        x, y, depth, self._scene.frame.width, self._scene.frame.height)
                    idx = self._calc_prefer_indicate(world_coord)
                    picked_point = np.asarray(self.mesh.vertices)[idx]
                    if self.draw_source_flag:
                        self.source_depth = depth
                print("pick point:", picked_point[0], picked_point[1], picked_point[2])

                if self.draw_source_flag:
                    self.source_pnt.append(picked_point)
                    self.draw_point(picked_point, rgb=(1, 0, 0), name="start" + str(len(self.source_pnt)))
                else:
                    self.target_pnt.append(picked_point)
                    self.draw_point(picked_point, rgb=(0, 0, 1), name="end" + str(len(self.target_pnt)))
                    self.draw_arrow(self.source_pnt[-1], self.target_pnt[-1], name='line' + str(len(self.source_pnt)))

                self.draw_source_flag = not self.draw_source_flag

            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _calc_prefer_indicate(self, point):
        [_, idx, _] = self.mesh_kdtree.search_knn_vector_3d(point, 1)
        return idx[0]

    def remove_geometry_name(self, name_list):
        if len(name_list) == 0:
            return

        def remove_geometry_name_main():
            for name in name_list:
                self._scene.scene.remove_geometry(name)
        gui.Application.instance.post_to_main_thread(self.window, remove_geometry_name_main)

    def draw_point(self, point, rgb=(0., 0., 0.), name='point'):
        def draw_point_main():
            sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.02)
            sphere.translate(point)
            material = rendering.MaterialRecord()
            material.shader = 'defaultUnlit'
            sphere.paint_uniform_color(rgb)
            self._scene.scene.add_geometry(name, sphere, material)
        gui.Application.instance.post_to_main_thread(self.window, draw_point_main)

    def draw_arrow(self, start_pnt, end_pnt, rgb=(0., 0.5, 0.), name='line'):
        def calculate_align_mat(vec):
            import math
            eps = 1e-8
            scale = np.linalg.norm(vec)
            vec = vec / scale
            z_unit_arr = np.array([0, 0, 1])
            if abs(np.dot(z_unit_arr, vec) + 1) < eps:
                trans_mat = -np.eye(3, 3)
            elif abs(np.dot(z_unit_arr, vec) - 1) < eps:
                trans_mat = np.eye(3, 3)
            else:
                cos_theta = np.dot(z_unit_arr, vec)
                rotate_axis = np.array([z_unit_arr[1] * vec[2] - z_unit_arr[2] * vec[1],
                                        z_unit_arr[2] * vec[0] - z_unit_arr[0] * vec[2],
                                        z_unit_arr[0] * vec[1] - z_unit_arr[1] * vec[0]])
                rotate_axis = rotate_axis / np.linalg.norm(rotate_axis)
                z_mat = np.array([[0, -rotate_axis[2], rotate_axis[1]],
                                  [rotate_axis[2], 0, -rotate_axis[0]],
                                  [-rotate_axis[1], rotate_axis[0], 0]])
                trans_mat = (np.eye(3, 3) + math.sin(math.acos(cos_theta)) * z_mat
                             + (1 - cos_theta) * np.matmul(z_mat, z_mat))
            return trans_mat

        direction = end_pnt - start_pnt
        rot_mat = calculate_align_mat(direction)
        cone_height = min(0.1, np.linalg.norm(direction) * 0.5)

        def draw_arrow_main():
            mesh_arrow = o3d.geometry.TriangleMesh().create_arrow(
                cone_height=cone_height,
                cone_radius=0.02,
                cylinder_height=np.linalg.norm(direction) - cone_height,
                cylinder_radius=0.01)
            mesh_arrow.paint_uniform_color(rgb)
            mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
            mesh_arrow.translate(self.source_pnt[-1])
            self._scene.scene.add_geometry(name, mesh_arrow, rendering.MaterialRecord())

        gui.Application.instance.post_to_main_thread(self.window, draw_arrow_main)

    def update_mesh(self, mesh, update_camera=True):
        if not isinstance(mesh, o3d.geometry.TriangleMesh):
            return

        def update_mesh_main():
            self.mesh_kdtree = o3d.geometry.KDTreeFlann(mesh)
            # mesh.compute_vertex_normals()
            # material = rendering.MaterialRecord()
            # material.shader = 'normals'
            mesh.compute_vertex_normals()
            material = rendering.MaterialRecord()
            material.shader = 'defaultLit'
            # material.base_color = [
            #     np.random.random(),
            #     np.random.random(),
            #     np.random.random(), 1.
            # ]
            material.base_color = [
                0.73,
                0.56,
                0.56, 1.
            ]
            if self._scene.scene.has_geometry('mesh'):
                self._scene.scene.remove_geometry('mesh')
            self._scene.scene.add_geometry("mesh", mesh, material)
            self.mesh = mesh
            if update_camera:
                bounds = mesh.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, mesh.get_center())

        gui.Application.instance.post_to_main_thread(self.window, update_mesh_main)

    def clear_all(self):
        self._scene.scene.clear_geometry()
        self.mesh = None
        self.source_pnt.clear()
        self.target_pnt.clear()
        self.draw_source_flag = True

    @staticmethod
    def run():
        gui.Application.instance.run()


if __name__ == "__main__":
    app = App()
    app.run()
