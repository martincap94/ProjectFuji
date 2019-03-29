#include "UserInterface.h"

#include <sstream>
#include <iomanip>

#include "TextureManager.h"
#include "Texture.h"
#include "LBM3D_1D_indices.h"
#include "ParticleSystem.h"
#include "Camera.h"
#include "DirectionalLight.h"
#include "EVSMShadowMapper.h"
#include "STLPDiagram.h"
#include "STLPSimulatorCUDA.h"
#include "ParticleRenderer.h"
#include "HosekSkyModel.h"
#include "Emitter.h"
#include "CircleEmitter.h"

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_GLFW_GL3_IMPLEMENTATION
#define NK_KEYSTATE_BASED_INPUT
#include <nuklear.h>
#include "nuklear_glfw_gl3.h"

#define INCLUDE_STYLE
#ifdef INCLUDE_STYLE
#include "nuklear/style.c"
#endif

#define MAX_VERTEX_BUFFER 512 * 1024
#define MAX_ELEMENT_BUFFER 128 * 1024

using namespace std;

UserInterface::UserInterface(GLFWwindow * window) {
	ctx = nk_glfw3_init(window, NK_GLFW3_INSTALL_CALLBACKS);


	{
		struct nk_font_atlas *atlas;
		nk_glfw3_font_stash_begin(&atlas);
		struct nk_font *roboto = nk_font_atlas_add_from_file(atlas, "nuklear/extra_font/Roboto-Regular.ttf", 14, 0);
		nk_glfw3_font_stash_end();
		nk_style_load_all_cursors(ctx, atlas->cursors);
		nk_style_set_font(ctx, &roboto->handle);
	}


#ifdef INCLUDE_STYLE
	set_style(ctx, THEME_MARTIN);
#endif
}

UserInterface::~UserInterface() {
}

void UserInterface::draw() {
	nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
}


void UserInterface::constructUserInterface() {
	nk_glfw3_new_frame();

	//ctx->style.window.padding = nk_vec2(10.0f, 10.0f);
	ctx->style.window.padding = nk_vec2(0.2f, 0.2f);

	stringstream ss;
	textures = TextureManager::getTexturesMapPtr();

	const struct nk_input *in = &ctx->input;
	struct nk_rect bounds;


	/* GUI */
	if (nk_begin(ctx, "Control Panel", nk_rect(0, vars->toolbarHeight, vars->leftSidebarWidth, vars->screenHeight - vars->debugTextureRes - vars->toolbarHeight),
				 NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR)) {


		nk_layout_row_dynamic(ctx, 30, 2);
		if (nk_button_label(ctx, "LBM")) {
			uiMode = 0;
		}
		if (nk_button_label(ctx, "Shadows")) {
			uiMode = 1;
		}
		if (nk_button_label(ctx, "Terrain")) {
			uiMode = 2;
		}
		if (nk_button_label(ctx, "Sky")) {
			uiMode = 3;
		}
		if (nk_button_label(ctx, "Cloud Visualization")) {
			uiMode = 4;
		}
		if (nk_button_label(ctx, "Diagram controls")) {
			uiMode = 5;
		}
		if (nk_button_label(ctx, "LBM DEVELOPER")) {
			uiMode = 6;
		}

		if (uiMode == 0) {
			constructLBMTab();
		} else if (uiMode == 1) {
			constructLightingTab();
		} else if (uiMode == 2) {
			constructTerrainTab();
		} else if (uiMode == 3) {
			constructSkyTab();
		} else if (uiMode == 4) {
			constructCloudVisualizationTab();
		} else if (uiMode == 5) {
			constructDiagramControlsTab();
		} else if (uiMode == 6) {
			constructLBMDebugTab();
		}




	}



	nk_end(ctx);


	if (nk_begin(ctx, "Debug Tab", nk_rect(vars->screenWidth - vars->rightSidebarWidth, vars->toolbarHeight, vars->rightSidebarWidth, /*vars->debugTabHeight*/vars->screenHeight - vars->toolbarHeight), NK_WINDOW_BORDER /*| NK_WINDOW_NO_SCROLLBAR*/)) {

		nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);

		/*
		ss.clear();
		ss << "Delta time: " << fixed << setprecision(2) << (deltaTime * 1000.0);

		//string fpsStr = "delta time: " + to_string(deltaTime * 1000.0);
		nk_label(ctx, ss.str().c_str(), NK_TEXT_CENTERED);
		*/
		stringstream().swap(ss);
		// TO DO - make this work
		//ss << "Delta time: " << fixed << setprecision(4) << prevAvgDeltaTime << " [ms] (" << setprecision(0) << prevAvgFPS << " FPS)";
		//nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);

		// Quick info -> creation of the strings should be moved to the Diagram since it only changes when the diagram is changed
		stringstream().swap(ss);
		ss << "(ref) T_c: " << fixed << setprecision(0) << stlpDiagram->Tc.x << " [deg C] at " << stlpDiagram->Tc.y << " [hPa]";
		nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);

		stringstream().swap(ss);
		ss << "CCL: " << fixed << setprecision(0) << stlpDiagram->CCL.x << " [deg C] at " << stlpDiagram->CCL.y << " [hPa]";

		nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);

		stringstream().swap(ss);
		ss << "EL: " << fixed << setprecision(0) << stlpDiagram->EL.x << " [deg C] at " << stlpDiagram->EL.y << " [hPa]";
		nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);

		stringstream().swap(ss);
		ss << "Ground pressure: " << stlpDiagram->P0 << " [hPa]";
		nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);


		vector<OverlayTexture *> *overlayTextures = TextureManager::getOverlayTexturesVectorPtr();

		for (int i = 0; i < overlayTextures->size(); i++) {

			if (nk_tree_push_id(ctx, NK_TREE_NODE, ("Overlay Texture " + to_string(i)).c_str(), NK_MAXIMIZED, i)) {
				nk_layout_row_dynamic(ctx, 15, 2);
				nk_checkbox_label(ctx, "#active", &(*overlayTextures)[i]->active);
				nk_checkbox_label(ctx, "#show alpha", &(*overlayTextures)[i]->showAlphaChannel);
				if (nk_combo_begin_label(ctx, (*overlayTextures)[i]->getBoundTextureName().c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
					nk_layout_row_dynamic(ctx, 15, 1);

					if (nk_combo_item_label(ctx, "NONE", NK_TEXT_CENTERED)) {
						(*overlayTextures)[i]->texture = nullptr;
					}
					for (const auto& kv : *textures) {
						if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_CENTERED)) {
							(*overlayTextures)[i]->texture = kv.second;
						}
					}
					nk_combo_end(ctx);
				}


				nk_tree_pop(ctx);
			}

		}

		nk_layout_row_dynamic(ctx, 15, 1);
		nk_label(ctx, "Camera info", NK_TEXT_CENTERED);

		nk_layout_row_dynamic(ctx, 15, 3);

		nk_value_float(ctx, "cam x:", camera->position.x);
		nk_value_float(ctx, "cam y:", camera->position.y);
		nk_value_float(ctx, "cam z:", camera->position.z);
		nk_value_float(ctx, "cam fx:", camera->front.x);
		nk_value_float(ctx, "cam fy:", camera->front.y);
		nk_value_float(ctx, "cam fz:", camera->front.z);

		nk_value_int(ctx, "w:", vars->latticeWidth);
		nk_value_int(ctx, "h:", vars->latticeHeight);
		nk_value_int(ctx, "d:", vars->latticeDepth);

		nk_value_float(ctx, "wr [m]", vars->latticeWidth * lbm->scale);
		nk_value_float(ctx, "wh [m]", vars->latticeHeight * lbm->scale);
		nk_value_float(ctx, "wd [m]", vars->latticeDepth * lbm->scale);

		nk_layout_row_dynamic(ctx, 15, 1);
		nk_value_int(ctx, "terrain width resolution", vars->heightMap->terrainWidth);
		nk_value_int(ctx, "terrain depth resolution", vars->heightMap->terrainDepth);




	}
	nk_end(ctx);








	ctx->style.window.padding = nk_vec2(0, 0);

	if (nk_begin(ctx, "test", nk_rect(0, 0, vars->screenWidth, vars->toolbarHeight), NK_WINDOW_NO_SCROLLBAR)) {

		int numToolbarItems = 3;

		//nk_layout_row_static(ctx, 32, screenWidth, numToolbarItems);
		//nk_menu

		/* menubar */
		enum menu_states { MENU_DEFAULT, MENU_WINDOWS };
		static nk_size mprog = 60;
		static int mslider = 10;
		static int mcheck = nk_true;
		nk_menubar_begin(ctx);

		/* menu #1 */
		nk_layout_row_begin(ctx, NK_STATIC, vars->toolbarHeight, 5);
		nk_layout_row_push(ctx, 120);
		if (nk_menu_begin_label(ctx, "File", NK_TEXT_CENTERED, nk_vec2(120, 200))) {
			static size_t prog = 40;
			static int slider = 10;
			static int check = nk_true;
			nk_layout_row_dynamic(ctx, 25, 1);
			nk_menu_item_label(ctx, "Hide", NK_TEXT_LEFT);
			nk_menu_item_label(ctx, "About", NK_TEXT_LEFT);
			nk_progress(ctx, &prog, 100, NK_MODIFIABLE);
			nk_slider_int(ctx, 0, &slider, 16, 1);
			nk_checkbox_label(ctx, "check", &check);

			if (nk_button_label(ctx, "EXIT")) {
				vars->appRunning = false;
			}
			nk_menu_end(ctx);
		}
		nk_layout_row_push(ctx, 120);
		//nk_label(ctx, "View", NK_TEXT_CENTERED);
		if (nk_menu_begin_label(ctx, "View", NK_TEXT_CENTERED, nk_vec2(120, 200))) {
			nk_layout_row_dynamic(ctx, 25, 1);
			//nk_button_label(ctx, "Debug Window");
			if (nk_menu_item_label(ctx, "Debug Window", NK_TEXT_CENTERED)) {
				cout << "opening debug window" << endl;
			}
			//struct nk_command_buffer* out = nk_window_get_canvas(ctx);
			//nk_stroke_line(out, )

			nk_label(ctx, "Camera Settings", NK_TEXT_CENTERED);
			if (nk_menu_item_label(ctx, "Front View (I)", NK_TEXT_CENTERED)) {
				camera->setView(Camera::VIEW_FRONT);
			}
			if (nk_menu_item_label(ctx, "Side View (O)", NK_TEXT_CENTERED)) {
				camera->setView(Camera::VIEW_SIDE);
			}
			if (nk_menu_item_label(ctx, "Top View (P)", NK_TEXT_CENTERED)) {
				camera->setView(Camera::VIEW_TOP);
			}


			if (vars->drawSkybox) {
				if (nk_menu_item_label(ctx, "Hide Skybox", NK_TEXT_CENTERED)) {
					vars->drawSkybox = false;
				}
			} else {
				if (nk_menu_item_label(ctx, "Show Skybox", NK_TEXT_CENTERED)) {
					vars->drawSkybox = true;
				}
			}
			

			nk_menu_end(ctx);

		}
		nk_layout_row_push(ctx, 120);
		if (nk_menu_begin_label(ctx, "About", NK_TEXT_CENTERED, nk_vec2(120, 200))) {
			nk_layout_row_dynamic(ctx, 25, 1);
			if (nk_menu_item_label(ctx, "Show About", NK_TEXT_CENTERED)) {
				vars->aboutWindowOpened = true;
			}

			nk_menu_end(ctx);
		}

		//nk_label(ctx, "About", NK_TEXT_CENTERED);

		//nk_layout_row_push(ctx, 120);
		//nk_label(ctx, "View", NK_TEXT_CENTERED);

		//nk_layout_row_push(ctx, 120);
		//nk_label(ctx, "View", NK_TEXT_CENTERED);
	}
	nk_end(ctx);


	/*

	static int show_group =  1;
	if (show_group) {
	nk_layout_row_dynamic(ctx, 100, 1);
	int res = nk_group_begin(ctx, "Node", NK_WINDOW_CLOSABLE|NK_WINDOW_BORDER);
	show_group = res != NK_WINDOW_CLOSED;
	if (res && show_group) {
	...
	nk_group_end(ctx);
	}
	}
	*/

	if (vars->aboutWindowOpened) {


		if (nk_begin(ctx, "About Window", nk_rect(vars->screenWidth / 2 - 250, vars->screenHeight / 2 - 250, 500, 500), NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_CLOSABLE)) {
			nk_layout_row_dynamic(ctx, 20.0f, 1);

			nk_label(ctx, "Orographic Cloud Simulator", NK_TEXT_CENTERED);
			nk_label(ctx, "Author: Martin Cap", NK_TEXT_CENTERED);
			nk_label(ctx, "Email: martincap94@gmail.com", NK_TEXT_CENTERED);


		} else {
			vars->aboutWindowOpened = false;
		}
		nk_end(ctx);


	}

	//nk_end(ctx);



}

bool UserInterface::isAnyWindowHovered() {
	return nk_window_is_any_hovered(ctx);
}

void UserInterface::uivec3(glm::vec3 & target) {

}

void UserInterface::uivec4(glm::vec4 & target) {

}

void UserInterface::uicolor(glm::vec4 & target) {

}

void UserInterface::constructLBMTab() {
	nk_layout_row_dynamic(ctx, 15, 1);
	nk_label(ctx, "LBM Controls", NK_TEXT_CENTERED);

	//if (nk_button_label(ctx, "fullscreen")) {
	//	vars->fullscreen = !vars->fullscreen;
	//	glfwWindowHint(GLFW_MAXIMIZED, vars->fullscreen ? GL_TRUE : GL_FALSE); // For maximization of window
	//}


	nk_layout_row_dynamic(ctx, 15, 1);
	if (nk_button_label(ctx, "Reset")) {
		//fprintf(stdout, "button pressed\n");
		lbm->resetSimulation();
		particleSystem->refreshParticlesOnTerrain();
	}
	const char *buttonDescription = vars->paused ? "Play" : "Pause";
	if (nk_button_label(ctx, buttonDescription)) {
		vars->paused = !vars->paused;
	}

	if (nk_button_label(ctx, "Refresh HEIGHTMAP")) {
		lbm->refreshHeightMap();
	}



	nk_property_float(ctx, "Tau:", 0.5005f, &lbm->tau, 10.0f, 0.005f, 0.005f);

	/*int mirrorSidesPrev = lbm->mirrorSides;
	nk_layout_row_dynamic(ctx, 15, 1);
	nk_checkbox_label(ctx, "Mirror sides", &lbm->mirrorSides);
	if (mirrorSidesPrev != lbm->mirrorSides) {
	cout << "Mirror sides value changed!" << endl;
	lbm->updateControlProperty(LBM3D_1D_indices::MIRROR_SIDES_PROP);
	}*/


	//nk_label(ctx, "Use point sprites", NK_TEXT_LEFT);
	int prevVsync = vars->vsync;
	nk_checkbox_label(ctx, "VSync", &vars->vsync);
	if (prevVsync != vars->vsync) {
		glfwSwapInterval(vars->vsync);
	}

	nk_label(ctx, "Inlet velocity:", NK_TEXT_LEFT);

	nk_property_float(ctx, "x:", -10.0f, &lbm->inletVelocity.x, 10.0f, 0.01f, 0.005f);
	nk_property_float(ctx, "y:", -10.0f, &lbm->inletVelocity.y, 10.0f, 0.01f, 0.005f);
	nk_property_float(ctx, "z:", -10.0f, &lbm->inletVelocity.z, 10.0f, 0.01f, 0.005f);



	//nk_label(ctx, "Use point sprites", NK_TEXT_LEFT);
	nk_checkbox_label(ctx, "Use point sprites", &vars->usePointSprites);


	if (nk_button_label(ctx, "Sort points by camera distance")) {
		particleSystem->sortParticlesByDistance(camera->position, eSortPolicy::GREATER);

	}
	if (/*lbmType == LBM2D &&*/ vars->useCUDA && !vars->usePointSprites) {
		nk_layout_row_dynamic(ctx, 15, 1);
		nk_checkbox_label(ctx, "Visualize velocity", &lbm->visualizeVelocity);
	}

	if (!vars->useCUDA) {
		nk_layout_row_dynamic(ctx, 15, 1);
		nk_checkbox_label(ctx, "Respawn linearly", &lbm->respawnLinearly);
	}
	/*
	nk_layout_row_dynamic(ctx, 10, 1);
	nk_labelf(ctx, NK_TEXT_LEFT, "Point size");
	nk_slider_float(ctx, 1.0f, &particleSystemLBM->pointSize, 100.0f, 0.5f);

	if (!vars->usePointSprites && !lbm->visualizeVelocity) {
	nk_layout_row_dynamic(ctx, 20, 1);
	nk_label(ctx, "Particles Color:", NK_TEXT_LEFT);
	nk_layout_row_dynamic(ctx, 25, 1);
	if (nk_combo_begin_color(ctx, nk_rgb_cf(particlesColor), nk_vec2(nk_widget_width(ctx), 400))) {
	nk_layout_row_dynamic(ctx, 120, 1);
	particlesColor = nk_color_picker(ctx, particlesColor, NK_RGBA);
	nk_layout_row_dynamic(ctx, 25, 1);
	particlesColor.r = nk_propertyf(ctx, "#R:", 0, particlesColor.r, 1.0f, 0.01f, 0.005f);
	particlesColor.g = nk_propertyf(ctx, "#G:", 0, particlesColor.g, 1.0f, 0.01f, 0.005f);
	particlesColor.b = nk_propertyf(ctx, "#B:", 0, particlesColor.b, 1.0f, 0.01f, 0.005f);
	particlesColor.a = nk_propertyf(ctx, "#A:", 0, particlesColor.a, 1.0f, 0.01f, 0.005f);
	particleSystemLBM->particlesColor = glm::vec3(particlesColor.r, particlesColor.g, particlesColor.b);
	nk_combo_end(ctx);
	}
	}*/
	nk_layout_row_dynamic(ctx, 15, 1);
	nk_label(ctx, "Camera movement speed", NK_TEXT_LEFT);
	nk_slider_float(ctx, 1.0f, &camera->movementSpeed, 10000.0f, 1.0f);


	// TO DO - get this back in working order
	/*


	nk_layout_row_dynamic(ctx, 15, 2);
	if (nk_option_label(ctx, "Orthographic", projectionMode == ORTHOGRAPHIC)) {
	projectionMode = ORTHOGRAPHIC;
	}
	if (nk_option_label(ctx, "Perspective", projectionMode == PERSPECTIVE)) {
	projectionMode = PERSPECTIVE;
	}
	if (projectionMode == PERSPECTIVE) {
	nk_slider_float(ctx, 30.0f, &fov, 120.0f, 1.0f);
	}



	//int useFreeRoamCameraPrev = vars->useFreeRoamCamera;
	nk_checkbox_label(ctx, "Use freeroam camera", &vars->useFreeRoamCamera);

	if (vars->useFreeRoamCamera) {
	FreeRoamCamera *fcam = (FreeRoamCamera*)camera;
	if (fcam) {
	nk_checkbox_label(ctx, "Walking", &fcam->walking);
	nk_property_float(ctx, "Player Height", 0.0f, &fcam->playerHeight, 10.0f, 0.01f, 0.01f);
	}


	}
	*/




	//if (useFreeRoamCameraPrev != vars->useFreeRoamCamera) {
	//	if (mode >= 2) {
	//		camera = (vars->useFreeRoamCamera) ? freeRoamCamera : viewportCamera;
	//		if (vars->useFreeRoamCamera) {
	//			cout << "using freeRoamCamera from now on" << endl;
	//		}
	//	}
	//}
	//nk_colorf()


	// TO DO - make this into a function
	struct nk_colorf tmpColor;
	tmpColor.r = dirLight->color.x;
	tmpColor.g = dirLight->color.y;
	tmpColor.b = dirLight->color.z;

	if (nk_combo_begin_color(ctx, nk_rgb_cf(tmpColor), nk_vec2(nk_widget_width(ctx), 400))) {
		nk_layout_row_dynamic(ctx, 120, 1);
		tmpColor = nk_color_picker(ctx, tmpColor, NK_RGBA);
		nk_layout_row_dynamic(ctx, 25, 1);
		tmpColor.r = nk_propertyf(ctx, "#R:", 0, tmpColor.r, 1.0f, 0.01f, 0.005f);
		tmpColor.g = nk_propertyf(ctx, "#G:", 0, tmpColor.g, 1.0f, 0.01f, 0.005f);
		tmpColor.b = nk_propertyf(ctx, "#B:", 0, tmpColor.b, 1.0f, 0.01f, 0.005f);
		tmpColor.a = nk_propertyf(ctx, "#A:", 0, tmpColor.a, 1.0f, 0.01f, 0.005f);
		dirLight->color = glm::vec3(tmpColor.r, tmpColor.g, tmpColor.b);
		nk_combo_end(ctx);
	}

	nk_layout_row_dynamic(ctx, 15, 1);
	nk_label(ctx, "LBM Respawn Mode", NK_TEXT_CENTERED);
	nk_layout_row_dynamic(ctx, 15, 2);
	if (nk_option_label(ctx, "Keep Position", lbm->respawnMode == LBM3D_1D_indices::KEEP_POSITION)) {
		lbm->respawnMode = LBM3D_1D_indices::KEEP_POSITION;
	}
	if (nk_option_label(ctx, "Random (Uniform)", lbm->respawnMode == LBM3D_1D_indices::RANDOM_UNIFORM)) {
		lbm->respawnMode = LBM3D_1D_indices::RANDOM_UNIFORM;
	}

	nk_layout_row_dynamic(ctx, 15, 1);
	nk_label(ctx, "LBM Out of Bounds Mode", NK_TEXT_CENTERED);
	nk_layout_row_dynamic(ctx, 15, 2);
	if (nk_option_label(ctx, "Ignore Particles", lbm->outOfBoundsMode == LBM3D_1D_indices::KEEP_POSITION)) {
		lbm->outOfBoundsMode = LBM3D_1D_indices::IGNORE_PARTICLES;
	}
	if (nk_option_label(ctx, "Deactivate Particles", lbm->outOfBoundsMode == LBM3D_1D_indices::DEACTIVATE_PARTICLES)) {
		lbm->outOfBoundsMode = LBM3D_1D_indices::DEACTIVATE_PARTICLES;
	}
	if (nk_option_label(ctx, "Respawn Particles in Inlet", lbm->outOfBoundsMode == LBM3D_1D_indices::RESPAWN_PARTICLES_INLET)) {
		lbm->outOfBoundsMode = LBM3D_1D_indices::RESPAWN_PARTICLES_INLET;
	}
	nk_layout_row_dynamic(ctx, 15, 1);
	nk_checkbox_label(ctx, "x left inlet", &lbm->xLeftInlet);
	nk_checkbox_label(ctx, "x right inlet", &lbm->xRightInlet);
	nk_checkbox_label(ctx, "y bottom inlet", &lbm->yBottomInlet);
	nk_checkbox_label(ctx, "y top inlet", &lbm->yTopInlet);
	nk_checkbox_label(ctx, "z left inlet", &lbm->zLeftInlet);
	nk_checkbox_label(ctx, "z right inlet", &lbm->zRightInlet);

	nk_checkbox_label(ctx, "Use subgrid model (experimental)", &vars->useSubgridModel);

	nk_property_float(ctx, "LBM velocity multiplier", 0.01f, &vars->lbmVelocityMultiplier, 10.0f, 0.01f, 0.01f);
	nk_checkbox_label(ctx, "LBM use correct interpolation", &vars->lbmUseCorrectInterpolation);
	nk_checkbox_label(ctx, "LBM use extended collision step", &vars->lbmUseExtendedCollisionStep);


}

void UserInterface::constructLightingTab() {


	nk_layout_row_dynamic(ctx, 30, 1);

	nk_label(ctx, "Lighting Controls", NK_TEXT_CENTERED);


	nk_layout_row_dynamic(ctx, 15, 1);
	nk_property_float(ctx, "#x:", -100000.0f, &dirLight->position.x, 100000.0f, 100.0f, 100.0f);
	nk_property_float(ctx, "#y:", -100000.0f, &dirLight->position.y, 100000.0f, 100.0f, 100.0f);
	nk_property_float(ctx, "#z:", -100000.0f, &dirLight->position.z, 100000.0f, 100.0f, 100.0f);


	nk_layout_row_dynamic(ctx, 15, 1);
	nk_property_float(ctx, "focus x:", -100000.0f, &dirLight->focusPoint.x, 100000.0f, 100.0f, 100.0f);
	nk_property_float(ctx, "focus y:", -100000.0f, &dirLight->focusPoint.y, 100000.0f, 100.0f, 100.0f);
	nk_property_float(ctx, "focus z:", -100000.0f, &dirLight->focusPoint.z, 100000.0f, 100.0f, 100.0f);


	nk_layout_row_dynamic(ctx, 15, 1);
	nk_property_float(ctx, "left:", -100000.0f, &dirLight->pLeft, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "right:", -100000.0f, &dirLight->pRight, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "bottom:", -100000.0f, &dirLight->pBottom, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "top:", -100000.0f, &dirLight->pTop, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "near:", 0.01f, &dirLight->pNear, 100.0f, 0.01f, 0.01f);
	nk_property_float(ctx, "far:", 1.0f, &dirLight->pFar, 100000.0f, 10.0f, 10.0f);

	nk_checkbox_label(ctx, "Simulate sun", &vars->simulateSun);
	nk_property_float(ctx, "Sun speed", 0.1f, &dirLight->circularMotionSpeed, 1000.0f, 0.1f, 0.1f);
	if (nk_option_label(ctx, "y axis", dirLight->rotationAxis == DirectionalLight::Y_AXIS)) {
		dirLight->rotationAxis = DirectionalLight::Y_AXIS;
	}
	if (nk_option_label(ctx, "z axis", dirLight->rotationAxis == DirectionalLight::Z_AXIS)) {
		dirLight->rotationAxis = DirectionalLight::Z_AXIS;
	}
	nk_property_float(ctx, "rotation radius:", 0.0f, &dirLight->radius, 10000.0f, 1.0f, 1.0f);

	nk_label(ctx, "EVSM", NK_TEXT_CENTERED);

	nk_checkbox_label(ctx, "use blur pass:", (int *)&evsm->useBlurPass);
	nk_property_float(ctx, "shadowBias:", 0.0f, &evsm->shadowBias, 1.0f, 0.0001f, 0.0001f);
	nk_property_float(ctx, "light bleed reduction:", 0.0f, &evsm->lightBleedReduction, 1.0f, 0.0001f, 0.0001f);
	//nk_property_float(ctx, "variance min limit:", 0.0f, &evsm.varianceMinLimit, 1.0f, 0.0001f, 0.0001f);
	nk_property_float(ctx, "exponent:", 1.0f, &evsm->exponent, 42.0f, 0.1f, 0.1f);

	nk_checkbox_label(ctx, "shadow only", &evsm->shadowOnly);


	nk_property_float(ctx, "Fog intensity", 0.0f, &vars->fogIntensity, 1.0f, 0.01f, 0.01f);

	if (nk_combo_begin_label(ctx, VariableManager::getFogModeString(vars->fogMode).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
		nk_layout_row_dynamic(ctx, 15, 1);
		if (nk_combo_item_label(ctx, "LINEAR", NK_TEXT_CENTERED)) {
			vars->fogMode = 0;
		}
		if (nk_combo_item_label(ctx, "EXPONENTIAL", NK_TEXT_CENTERED)) {
			vars->fogMode = 1;
		}
		nk_combo_end(ctx);

	}

	/*if (nk_combo_begin_label(ctx, "", nk_vec2(nk_widget_width(ctx), 200))) {
	nk_layout_row_dynamic(ctx, 15, 1);
	if (nk_combo_item_label(ctx, "NONE", NK_TEXT_CENTERED)) {
	vars->heightMap->materials[i].diffuseTexture = nullptr;
	}
	for (const auto& kv : *textures) {
	if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_CENTERED)) {
	vars->heightMap->materials[i].diffuseTexture = kv.second;
	}
	}
	nk_combo_end(ctx);
	}*/

	if (vars->fogMode == VariableManager::eFogMode::LINEAR) {
		nk_property_float(ctx, "Fog min distance", 0.0f, &vars->fogMinDistance, 1000.0f, 0.1f, 0.1f);
		nk_property_float(ctx, "Fog max distance", 0.0f, &vars->fogMaxDistance, 100000.0f, 100.0f, 100.0f);
	} else {
		nk_property_float(ctx, "Fog exp falloff", 0.0f, &vars->fogExpFalloff, 1.0f, 0.01f, 0.01f);
	}

	struct nk_colorf tmpColor;
	tmpColor.r = vars->fogColor.x;
	tmpColor.g = vars->fogColor.y;
	tmpColor.b = vars->fogColor.z;
	tmpColor.a = vars->fogColor.w;

	if (nk_combo_begin_color(ctx, nk_rgb_cf(tmpColor), nk_vec2(nk_widget_width(ctx), 400))) {
		nk_layout_row_dynamic(ctx, 120, 1);
		tmpColor = nk_color_picker(ctx, tmpColor, NK_RGBA);
		nk_layout_row_dynamic(ctx, 25, 1);
		tmpColor.r = nk_propertyf(ctx, "#R:", 0, tmpColor.r, 1.0f, 0.01f, 0.005f);
		tmpColor.g = nk_propertyf(ctx, "#G:", 0, tmpColor.g, 1.0f, 0.01f, 0.005f);
		tmpColor.b = nk_propertyf(ctx, "#B:", 0, tmpColor.b, 1.0f, 0.01f, 0.005f);
		tmpColor.a = nk_propertyf(ctx, "#A:", 0, tmpColor.a, 1.0f, 0.01f, 0.005f);
		vars->fogColor = glm::vec4(tmpColor.r, tmpColor.g, tmpColor.b, tmpColor.a);
		nk_combo_end(ctx);
	}


	//nk_checkbox_label(ctx, "Run Harris 1st pass in next frame", &vars->run_harris_1st_pass_inNextFrame);
	if (nk_button_label(ctx, "Run Harris 1st pass in the next frame")) {
		vars->run_harris_1st_pass_inNextFrame = 1;
	}



}

void UserInterface::constructTerrainTab() {



	nk_layout_row_dynamic(ctx, 30, 1);

	nk_label(ctx, "Terrain Controls", NK_TEXT_CENTERED);

	nk_layout_row_dynamic(ctx, 15, 1);

	nk_checkbox_label(ctx, "Draw grass", &vars->drawGrass);
	nk_checkbox_label(ctx, "Draw trees", &vars->drawTrees);
	nk_checkbox_label(ctx, "Visualize normals", &vars->visualizeTerrainNormals);
	nk_property_float(ctx, "Nrm mixing ratio: ", 0.0f, &vars->globalNormalMapMixingRatio, 1.0f, 0.01f, 0.01f);

	for (int i = 0; i < MAX_TERRAIN_MATERIALS; i++) {

		if (nk_tree_push_id(ctx, NK_TREE_NODE, ("Material " + to_string(i)).c_str(), NK_MAXIMIZED, i)) {

			nk_layout_row_dynamic(ctx, 15, 2);

			nk_label(ctx, "Diffuse", NK_TEXT_LEFT);

			if (nk_combo_begin_label(ctx, vars->heightMap->materials[i].tryGetTextureFilename(Texture::eTextureMaterialType::DIFFUSE).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
				nk_layout_row_dynamic(ctx, 15, 1);
				if (nk_combo_item_label(ctx, "NONE", NK_TEXT_CENTERED)) {
					vars->heightMap->materials[i].diffuseTexture = nullptr;
				}
				for (const auto& kv : *textures) {
					if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_CENTERED)) {
						vars->heightMap->materials[i].diffuseTexture = kv.second;
					}
				}
				nk_combo_end(ctx);
			}

			nk_label(ctx, "Specular", NK_TEXT_LEFT);

			if (nk_combo_begin_label(ctx, vars->heightMap->materials[i].tryGetTextureFilename(Texture::eTextureMaterialType::SPECULAR).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
				nk_layout_row_dynamic(ctx, 15, 1);
				if (nk_combo_item_label(ctx, "NONE", NK_TEXT_CENTERED)) {
					vars->heightMap->materials[i].specularMap = nullptr;
				}
				for (const auto& kv : *textures) {
					if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_CENTERED)) {
						vars->heightMap->materials[i].specularMap = kv.second;
					}
				}
				nk_combo_end(ctx);
			}

			nk_label(ctx, "Normal Map", NK_TEXT_LEFT);

			if (nk_combo_begin_label(ctx, vars->heightMap->materials[i].tryGetTextureFilename(Texture::eTextureMaterialType::NORMAL_MAP).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
				nk_layout_row_dynamic(ctx, 15, 1);
				if (nk_combo_item_label(ctx, "NONE", NK_TEXT_CENTERED)) {
					vars->heightMap->materials[i].normalMap = nullptr;
				}
				for (const auto& kv : *textures) {
					if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_CENTERED)) {
						vars->heightMap->materials[i].normalMap = kv.second;
					}
				}
				nk_combo_end(ctx);
			}

			nk_layout_row_dynamic(ctx, 15, 1);
			nk_property_float(ctx, "#shininess", 0.0f, &vars->heightMap->materials[i].shininess, 128.0f, 0.1f, 0.1f);
			nk_property_float(ctx, "#tiling", 0.1f, &vars->heightMap->materials[i].textureTiling, 1000.0f, 0.1f, 1.0f);


			nk_tree_pop(ctx);
		}

	}



	if (nk_button_label(ctx, "Refresh LBM HEIGHTMAP")) {
		lbm->refreshHeightMap();
	}
	nk_property_int(ctx, "x offset", 0, &vars->terrainXOffset, 1000, 1, 1);
	nk_property_int(ctx, "z offset", 0, &vars->terrainZOffset, 1000, 1, 1);


}

void UserInterface::constructSkyTab() {

	nk_layout_row_dynamic(ctx, 30, 1);

	nk_label(ctx, "Sky", NK_TEXT_CENTERED);


	nk_layout_row_dynamic(ctx, 15, 1);
	nk_property_float(ctx, "#x:", -1000.0f, &dirLight->position.x, 1000.0f, 1.0f, 1.0f);
	nk_property_float(ctx, "#y:", -1000.0f, &dirLight->position.y, 1000.0f, 1.0f, 1.0f);
	nk_property_float(ctx, "#z:", -1000.0f, &dirLight->position.z, 1000.0f, 1.0f, 1.0f);


	nk_checkbox_label(ctx, "Skybox", &vars->drawSkybox);
	nk_checkbox_label(ctx, "Hosek", &vars->hosekSkybox);


	nk_property_float(ctx, "Turbidity", 1.0f, &hosek->turbidity, 10.0f, 0.1f, 0.1f);
	nk_property_float(ctx, "Albedo", 0.0f, &hosek->albedo, 1.0f, 0.01f, 0.01f);


	nk_property_float(ctx, "Horizon Offset", 0.001f, &hosek->horizonOffset, 10.0f, 0.001f, 0.001f);


	nk_checkbox_label(ctx, "Recalculate Live", &hosek->liveRecalc);



	if (!hosek->liveRecalc) {
		if (nk_button_label(ctx, "Recalculate Model")) {
			hosek->update(dirLight->getDirection());

		}
	}



	nk_value_float(ctx, "Eta", hosek->eta);
	nk_value_float(ctx, "Eta (degrees)", hosek->getElevationDegrees());




}

void UserInterface::constructCloudVisualizationTab() {

	nk_layout_row_dynamic(ctx, 30, 1);

	nk_label(ctx, "Cloud Visualization", NK_TEXT_CENTERED);


	nk_layout_row_dynamic(ctx, 15, 1);
	nk_property_float(ctx, "#x:", -1000.0f, &dirLight->position.x, 1000.0f, 1.0f, 1.0f);
	nk_property_float(ctx, "#y:", -1000.0f, &dirLight->position.y, 1000.0f, 1.0f, 1.0f);
	nk_property_float(ctx, "#z:", -1000.0f, &dirLight->position.z, 1000.0f, 1.0f, 1.0f);


	nk_layout_row_dynamic(ctx, 15, 1);

	int prevNumSlices = particleRenderer->numSlices;
	nk_property_int(ctx, "Num slices", 1, &particleRenderer->numSlices, particleRenderer->maxNumSlices, 1, 1);
	if (prevNumSlices != particleRenderer->numSlices) {
		particleRenderer->numDisplayedSlices = particleRenderer->numSlices;
	}

	nk_property_int(ctx, "Num displayed slices", 0, &particleRenderer->numDisplayedSlices, particleRenderer->numSlices, 1, 1);

	nk_value_int(ctx, "Batch size", particleRenderer->batchSize);


	nk_checkbox_label(ctx, "Draw volume particles", &vars->renderVolumeParticlesDirectly);


	if (nk_button_label(ctx, "Form BOX")) {
		particleSystem->formBox(glm::vec3(20.0f), glm::vec3(20.0f));
	}


	nk_property_float(ctx, "Shadow alpha (100x)", 0.01f, &particleRenderer->shadowAlpha100x, 100.0f, 0.01f, 0.01f);



	nk_value_bool(ctx, "Inverted rendering", particleRenderer->invertedView);




	nk_property_float(ctx, "Point size", 0.1f, &stlpSimCUDA->pointSize, 100000.0f, 0.1f, 0.1f);
	particleSystem->pointSize = stlpSimCUDA->pointSize;
	nk_property_float(ctx, "Opacity multiplier", 0.01f, &vars->opacityMultiplier, 10.0f, 0.01f, 0.01f);


	nk_checkbox_label(ctx, "use new", &particleRenderer->compositeResultToFramebuffer);

	nk_property_int(ctx, "first pass shader mode", 0, &particleRenderer->firstPassShaderMode, particleRenderer->numFirstPassShaderModes - 1, 1, 1);

	nk_property_int(ctx, "second pass shader mode", 0, &particleRenderer->secondPassShaderMode, particleRenderer->numSecondPassShaderModes - 1, 1, 1);



	if (nk_combo_begin_label(ctx, getTextureName(particleRenderer->spriteTexture).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
		nk_layout_row_dynamic(ctx, 15, 1);

		/*
		if (nk_combo_item_label(ctx, "NONE", NK_TEXT_CENTERED)) {
		particleRenderer->spriteTexture = nullptr; // not a very bright idea
		}
		*/
		for (int i = 0; i < particleRenderer->spriteTextures.size(); i++) {
			if (nk_combo_item_label(ctx, particleRenderer->spriteTextures[i]->filename.c_str(), NK_TEXT_CENTERED)) {
				particleRenderer->spriteTexture = particleRenderer->spriteTextures[i];
			}
		}
		nk_combo_end(ctx);
	}
}

void UserInterface::constructDiagramControlsTab() {




	nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);
	if (nk_button_label(ctx, "Recalculate Params")) {
		//lbm->resetSimulation();
		stlpDiagram->recalculateParameters();
		stlpSimCUDA->uploadDataFromDiagramToGPU();
	}



	if (nk_tree_push(ctx, NK_TREE_TAB, "Diagram controls", NK_MINIMIZED)) {
		nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);

		nk_checkbox_label(ctx, "Show isobars", &stlpDiagram->showIsobars);
		nk_checkbox_label(ctx, "Show isotherms", &stlpDiagram->showIsotherms);
		nk_checkbox_label(ctx, "Show isohumes", &stlpDiagram->showIsohumes);
		nk_checkbox_label(ctx, "Show dry adiabats", &stlpDiagram->showDryAdiabats);
		nk_checkbox_label(ctx, "Show moist adiabats", &stlpDiagram->showMoistAdiabats);
		nk_checkbox_label(ctx, "Show dewpoint curve", &stlpDiagram->showDewpointCurve);
		nk_checkbox_label(ctx, "Show ambient temp. curve", &stlpDiagram->showAmbientTemperatureCurve);
		nk_checkbox_label(ctx, "Crop Bounds", &stlpDiagram->cropBounds);

		nk_tree_pop(ctx);

	}

	nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);

	int tmp = stlpDiagram->overlayDiagramWidth;
	int maxDiagramWidth = (vars->screenWidth < vars->screenHeight) ? vars->screenWidth : vars->screenHeight;
	nk_slider_int(ctx, 10, (int *)&stlpDiagram->overlayDiagramWidth, maxDiagramWidth, 1);
	if (tmp != stlpDiagram->overlayDiagramWidth) {
		stlpDiagram->overlayDiagramHeight = stlpDiagram->overlayDiagramWidth;
		stlpDiagram->refreshOverlayDiagram(vars->screenWidth, vars->screenHeight);
	}

	if (nk_button_label(ctx, "Reset to default")) {
		stlpDiagram->resetToDefault();
	}

	//if (nk_button_label(ctx, "Reset simulation")) {
	//stlpSim->resetSimulation();
	//}

	//nk_slider_float(ctx, 0.01f, &stlpSim->simulationSpeedMultiplier, 1.0f, 0.01f);

	float delta_t_prev = stlpSimCUDA->delta_t;
	nk_property_float(ctx, "delta t", 0.001f, &stlpSimCUDA->delta_t, 100.0f, 0.001f, 0.001f);
	if (stlpSimCUDA->delta_t != delta_t_prev) {
		//stlpSimCUDA->delta_t = stlpSim->delta_t;
		stlpSimCUDA->updateGPU_delta_t();
	}

	nk_property_int(ctx, "number of profiles", 2, &stlpDiagram->numProfiles, 100, 1, 1.0f); // somewhere bug when only one profile -> FIX!

	nk_property_float(ctx, "profile range", -10.0f, &stlpDiagram->convectiveTempRange, 10.0f, 0.01f, 0.01f);

	//nk_property_int(ctx, "max particles", 1, &stlpSim->maxNumParticles, 100000, 1, 10.0f);

	//nk_checkbox_label(ctx, "Simulate wind", &stlpSim->simulateWind);

	//nk_checkbox_label(ctx, "use prev velocity", &stlpSim->usePrevVelocity);

	nk_checkbox_label(ctx, "Divide Previous Velocity", &vars->dividePrevVelocity);
	if (vars->dividePrevVelocity) {
		nk_property_float(ctx, "Divisor (x100)", 100.0f, &vars->prevVelocityDivisor, 1000.0f, 0.1f, 0.1f); // [1.0, 10.0]
	}

	nk_checkbox_label(ctx, "Show CCL Level", &vars->showCCLLevelLayer);
	nk_checkbox_label(ctx, "Show EL Level", &vars->showELLevelLayer);
	nk_checkbox_label(ctx, "Show Overlay Diagram", &vars->showOverlayDiagram);


	nk_checkbox_label(ctx, "Use CUDA", &vars->stlpUseCUDA);

	nk_checkbox_label(ctx, "Apply LBM", &vars->applyLBM);
	nk_property_int(ctx, "LBM step frame", 1, &vars->lbmStepFrame, 100, 1, 1);

	/*
	bounds = nk_widget_bounds(ctx);
	if (nk_input_is_mouse_hovering_rect(in, bounds)) {
	nk_tooltip(ctx, "This is a tooltip");
	}
	*/



	nk_checkbox_label(ctx, "Apply STLP", &vars->applySTLP);
	nk_property_int(ctx, "STLP step frame", 1, &vars->stlpStepFrame, 100, 1, 1);

	nk_property_float(ctx, "Point size", 0.1f, &stlpSimCUDA->pointSize, 100.0f, 0.1f, 0.1f);
	//stlpSimCUDA->pointSize = stlpSim->pointSize;
	particleSystem->pointSize = stlpSimCUDA->pointSize;
	//nk_property_float(ctx, "Point size (CUDA)", 0.1f, &stlpSimCUDA->pointSize, 100.0f, 0.1f, 0.1f);

	nk_property_float(ctx, "Opacity multiplier", 0.01f, &vars->opacityMultiplier, 10.0f, 0.01f, 0.01f);
	nk_checkbox_label(ctx, "Show Cloud Shadows", &vars->showCloudShadows);

	struct nk_colorf tintColor;
	tintColor.r = vars->tintColor.x;
	tintColor.g = vars->tintColor.y;
	tintColor.b = vars->tintColor.z;

	if (nk_combo_begin_color(ctx, nk_rgb_cf(tintColor), nk_vec2(nk_widget_width(ctx), 400))) {
		nk_layout_row_dynamic(ctx, 120, 1);
		tintColor = nk_color_picker(ctx, tintColor, NK_RGBA);
		nk_layout_row_dynamic(ctx, 10, 1);
		tintColor.r = nk_propertyf(ctx, "#R:", 0, tintColor.r, 1.0f, 0.01f, 0.005f);
		tintColor.g = nk_propertyf(ctx, "#G:", 0, tintColor.g, 1.0f, 0.01f, 0.005f);
		tintColor.b = nk_propertyf(ctx, "#B:", 0, tintColor.b, 1.0f, 0.01f, 0.005f);
		tintColor.a = nk_propertyf(ctx, "#A:", 0, tintColor.a, 1.0f, 0.01f, 0.005f);
		vars->tintColor = glm::vec3(tintColor.r, tintColor.g, tintColor.b);
		nk_combo_end(ctx);
	}



	for (int i = 0; i < particleSystem->emitters.size(); i++) {
		if (nk_tree_push_id(ctx, NK_TREE_NODE, ("#Emitter " + to_string(i)).c_str(), NK_MINIMIZED, i)) {
			Emitter *e = particleSystem->emitters[i];

			nk_layout_row_static(ctx, 15, 200, 1);
			nk_checkbox_label(ctx, "#enabled", &e->enabled);
			nk_checkbox_label(ctx, "#visible", &e->visible);
			nk_checkbox_label(ctx, "#wiggle", &e->wiggle);
			nk_property_float(ctx, "#x wiggle", 0.1f, &e->xWiggleRange, 10.0f, 0.1f, 0.1f);
			nk_property_float(ctx, "#z wiggle", 0.1f, &e->zWiggleRange, 10.0f, 0.1f, 0.1f);


			nk_property_float(ctx, "#x", -1000.0f, &e->position.x, 1000.0f, 1.0f, 1.0f);
			//nk_property_float(ctx, "#y", -1000.0f, &e->position.y, 1000.0f, 1.0f, 1.0f);
			nk_property_float(ctx, "#z", -1000.0f, &e->position.z, 1000.0f, 1.0f, 1.0f);

			//nk_property_variant_int()
			nk_property_int(ctx, "#emit per step", 0, &e->numParticlesToEmitPerStep, 10000, 10, 10);

			CircleEmitter *ce = dynamic_cast<CircleEmitter *>(e);
			if (ce) {
				nk_property_float(ctx, "#radius", 1.0f, &ce->radius, 1000.0f, 1.0f, 1.0f);
			}


			nk_tree_pop(ctx);
			//particleSystem->emitters[i]
		}
	}
	nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);
	if (nk_button_label(ctx, "Activate All Particles")) {
		particleSystem->activateAllParticles();
		vars->run_harris_1st_pass_inNextFrame = 1; // DEBUG
	}
	if (nk_button_label(ctx, "Deactivate All Particles")) {
		particleSystem->deactivateAllParticles();
	}
	if (nk_button_label(ctx, "Enable All Emitters")) {
		particleSystem->enableAllEmitters();
	}
	if (nk_button_label(ctx, "Disable All Emitters")) {
		particleSystem->disableAllEmitters();
	}

	nk_property_int(ctx, "Active Particles", 0, &particleSystem->numActiveParticles, particleSystem->numParticles, 1000, 100);



	tintColor.r = vars->bgClearColor.x;
	tintColor.g = vars->bgClearColor.y;
	tintColor.b = vars->bgClearColor.z;

	if (nk_combo_begin_color(ctx, nk_rgb_cf(tintColor), nk_vec2(nk_widget_width(ctx), 400))) {
		nk_layout_row_dynamic(ctx, 120, 1);
		tintColor = nk_color_picker(ctx, tintColor, NK_RGBA);
		nk_layout_row_dynamic(ctx, 10, 1);
		tintColor.r = nk_propertyf(ctx, "#R:", 0, tintColor.r, 1.0f, 0.01f, 0.005f);
		tintColor.g = nk_propertyf(ctx, "#G:", 0, tintColor.g, 1.0f, 0.01f, 0.005f);
		tintColor.b = nk_propertyf(ctx, "#B:", 0, tintColor.b, 1.0f, 0.01f, 0.005f);
		//tintColor.a = nk_propertyf(ctx, "#A:", 0, tintColor.a, 1.0f, 0.01f, 0.005f);
		vars->bgClearColor = glm::vec3(tintColor.r, tintColor.g, tintColor.b);
		nk_combo_end(ctx);
	}
}

void UserInterface::constructLBMDebugTab() {


	nk_layout_row_dynamic(ctx, 30, 1);
	nk_label(ctx, "LBM DEVELOPER", NK_TEXT_CENTERED);

	nk_layout_row_dynamic(ctx, 15, 1);


	if (!lbm->isUnderEdit()) {
		if (nk_button_label(ctx, "EDIT LBM")) {
			lbm->startEditing();
		}
	} else {

		nk_property_float(ctx, "#x", -10000.0f, &lbm->position.x, 100000.0f, 10.0f, 10.0f);
		nk_property_float(ctx, "#y", -10000.0f, &lbm->position.y, 100000.0f, 10.0f, 10.0f);
		nk_property_float(ctx, "#z", -10000.0f, &lbm->position.z, 100000.0f, 10.0f, 10.0f);

		/*
		// This needs LBM reinitialization - maybe later
		nk_property_int(ctx, "x cells", 10, &lbm->latticeWidth, 1000, 1, 1);
		nk_property_int(ctx, "y cells", 10, &lbm->latticeHeight, 1000, 1, 1);
		nk_property_int(ctx, "z cells", 10, &lbm->latticeDepth, 1000, 1, 1);
		*/

		nk_property_float(ctx, "scale", 1.0f, &lbm->scale, 1000.0f, 1.0f, 1.0f);

		if (nk_button_label(ctx, "SAVE CHANGES")) {

			//lbm->saveChanges(); // testing 
			lbm->stopEditing(true);
		}
		if (nk_button_label(ctx, "CANCEL")) {
			lbm->stopEditing(false);
		}

	}
}

void UserInterface::constructDebugTab() {
}

void UserInterface::uivec2(glm::vec2 & target) {

}