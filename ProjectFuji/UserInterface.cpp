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
#include "FreeRoamCamera.h"
#include "StreamlineParticleSystem.h"
#include "Utils.h"
#include "PositionalEmitter.h"
#include "SceneGraph.h"

#define NK_IMPLEMENTATION
#define NK_GLFW_GL3_IMPLEMENTATION
#include <nuklear.h> // this is mandatory since we need the NK_IMPLEMENTATION


#include "nuklear_glfw_gl3.h"


#define INCLUDE_STYLE
#ifdef INCLUDE_STYLE
#include "nuklear/style.c"
#endif


using namespace std;

UserInterface::UserInterface(GLFWwindow *window, VariableManager *vars) : vars(vars) {
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

	editIcon = TextureManager::loadTexture("icons/edit.png");
	settingsIcon = TextureManager::loadTexture("icons/settings.png");

	nkEditIcon = nk_image_id(editIcon->id);
	nkSettingsIcon = nk_image_id(settingsIcon->id);

	leftSidebarWidth = (float)vars->leftSidebarWidth + leftSidebarBorderWidth;

	ctx_in = &ctx->input;

	ctx->style.button.rounding = 0.0f;
	ctx->style.button.image_padding = nk_vec2(0.0f, 0.0f);

	leftSidebarEditButtonRatio[0] = leftSidebarWidth - 20.0f;
	leftSidebarEditButtonRatio[1] = 15.0f;

}

UserInterface::~UserInterface() {
}

void UserInterface::draw() {
	nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
}


void UserInterface::constructUserInterface() {
	if (vars->hideUI) {
		return;
	}

	nk_glfw3_new_frame();

	//ctx->style.window.padding = nk_vec2(10.0f, 10.0f);
	ctx->style.window.padding = nk_vec2(0.2f, 0.2f);

	textures = TextureManager::getTexturesMapPtr();

	const struct nk_input *in = &ctx->input;
	struct nk_rect bounds;

	//ctx->style.window.border = 5.0f;

	//ctx->style.tab.rounding = 0.0f;
	//ctx->style.button.rounding = 0.0f;
	ctx->style.property.rounding = 0.0f;

	constructLeftSidebar();
	constructRightSidebar();

	constructHorizontalBar();

	constructTerrainGeneratorWindow();

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


}

bool UserInterface::isAnyWindowHovered() {
	return nk_window_is_any_hovered(ctx);
}

void UserInterface::nk_property_vec2(nk_context * ctx, float min, glm::vec2 & target, float max, float step, float pixStep, std::string label, eVecNaming namingConvention) {
	if (!label.empty()) {
		nk_label(ctx, label.c_str(), NK_TEXT_CENTERED);
	}

	int idxOffset = namingConvention * 4;
	for (int i = 0; i < 2; i++) {
		nk_property_float(ctx, vecNames[i + idxOffset], min, &target[i], max, step, pixStep);
	}
}

void UserInterface::nk_property_vec3(nk_context * ctx, float min, glm::vec3 & target, float max, float step, float pixStep, std::string label, eVecNaming namingConvention) {
	if (!label.empty()) {
		nk_label(ctx, label.c_str(), NK_TEXT_CENTERED);
	}

	int idxOffset = namingConvention * 4;
	for (int i = 0; i < 3; i++) {
		nk_property_float(ctx, vecNames[i + idxOffset], min, &target[i], max, step, pixStep);
	}
}

void UserInterface::nk_property_vec4(nk_context * ctx, glm::vec4 & target) {
	REPORT_NOT_IMPLEMENTED();
}

void UserInterface::nk_property_color_rgb(nk_context * ctx, glm::vec3 & target) {
	struct nk_colorf tmp;
	tmp.r = target.r;
	tmp.g = target.g;
	tmp.b = target.b;
	if (nk_combo_begin_color(ctx, nk_rgb_cf(tmp), nk_vec2(nk_widget_width(ctx), 400))) {
		nk_layout_row_dynamic(ctx, 120, 1);
		tmp = nk_color_picker(ctx, tmp, NK_RGB);
		nk_layout_row_dynamic(ctx, 15, 1);
		target.r = nk_propertyf(ctx, "#R:", 0, tmp.r, 1.0f, 0.01f, 0.005f);
		target.g = nk_propertyf(ctx, "#G:", 0, tmp.g, 1.0f, 0.01f, 0.005f);
		target.b = nk_propertyf(ctx, "#B:", 0, tmp.b, 1.0f, 0.01f, 0.005f);
		nk_combo_end(ctx);
	}
}

void UserInterface::nk_property_color_rgba(nk_context * ctx, glm::vec4 & target) {
	struct nk_colorf tmp;
	tmp.r = target.r;
	tmp.g = target.g;
	tmp.b = target.b;
	tmp.a = target.a;
	if (nk_combo_begin_color(ctx, nk_rgb_cf(tmp), nk_vec2(nk_widget_width(ctx), 400))) {
		nk_layout_row_dynamic(ctx, 120, 1);
		tmp = nk_color_picker(ctx, tmp, NK_RGBA);
		nk_layout_row_dynamic(ctx, 15, 1);
		target.r = nk_propertyf(ctx, "#R:", 0, tmp.r, 1.0f, 0.01f, 0.005f);
		target.g = nk_propertyf(ctx, "#G:", 0, tmp.g, 1.0f, 0.01f, 0.005f);
		target.b = nk_propertyf(ctx, "#B:", 0, tmp.b, 1.0f, 0.01f, 0.005f);
		target.a = nk_propertyf(ctx, "#A:", 0, tmp.a, 1.0f, 0.01f, 0.005f);
		nk_combo_end(ctx);
	}
}

void UserInterface::nk_value_vec3(nk_context * ctx, const glm::vec3 & target, std::string label, eVecNaming namingConvention) {
	if (!label.empty()) {
		nk_label(ctx, label.c_str(), NK_TEXT_CENTERED);
	}
	nk_layout_row_dynamic(ctx, 15, 3);

	int idxOffset = namingConvention * 4;
	for (int i = 0; i < 3; i++) {
		nk_value_float(ctx, vecNames[i + idxOffset], target[i]);
	}
	nk_layout_row_dynamic(ctx, 15, 1);

}



void UserInterface::constructLeftSidebar() {


	ctx->style.window.border = leftSidebarBorderWidth;

	ctx->style.button.padding = nk_vec2(0.0f, 0.0f);
	ctx->style.button.border = 0.0f;

	constructSidebarSelectionTab(&leftSidebarContentMode, 0, leftSidebarWidth + 20.0f);

	if (nk_begin(ctx, "Control Panel", nk_rect(0, vars->toolbarHeight + selectionTabHeight, leftSidebarWidth + 20.0f, vars->screenHeight - vars->toolbarHeight - selectionTabHeight),
				 NK_WINDOW_BORDER /*| NK_WINDOW_NO_SCROLLBAR*/)) {
		constructSelectedContent(leftSidebarContentMode);

	}
	nk_end(ctx);


}

void UserInterface::constructRightSidebar() {

	constructSidebarSelectionTab(&rightSidebarContentMode, vars->screenWidth - vars->rightSidebarWidth, vars->rightSidebarWidth);


	if (nk_begin(ctx, "Debug Tab", nk_rect(vars->screenWidth - vars->rightSidebarWidth, vars->toolbarHeight + selectionTabHeight, vars->rightSidebarWidth, /*vars->debugTabHeight*/vars->screenHeight - vars->toolbarHeight - selectionTabHeight), NK_WINDOW_BORDER /*| NK_WINDOW_NO_SCROLLBAR*/)) {

		constructSelectedContent(rightSidebarContentMode);
		//constructGeneralDebugTab();

	}
	nk_end(ctx);
}

void UserInterface::constructHorizontalBar() {

	ctx->style.window.padding = nk_vec2(0, 0);

	if (nk_begin(ctx, "HorizontalBar", nk_rect(0, 0, vars->screenWidth, vars->toolbarHeight), NK_WINDOW_NO_SCROLLBAR)) {

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
			nk_layout_row_dynamic(ctx, 15, 1);

			nk_checkbox_label(ctx, "Render Mode", &vars->renderMode);

			if (viewportMode == eViewportMode::VIEWPORT_3D) {
				if (nk_menu_item_label(ctx, "Diagram view", NK_TEXT_CENTERED)) {
					viewportMode == eViewportMode::DIAGRAM;
				}
			} else if (viewportMode == eViewportMode::DIAGRAM) {
				if (nk_menu_item_label(ctx, "3D viewport", NK_TEXT_CENTERED)) {
					viewportMode = eViewportMode::VIEWPORT_3D;
				}
			}


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
			nk_layout_row_dynamic(ctx, 15.0f, 1);
			if (nk_menu_item_label(ctx, "Show About", NK_TEXT_CENTERED)) {
				vars->aboutWindowOpened = true;
			}

			nk_menu_end(ctx);
		}
		constructFavoritesMenu();
		//nk_label(ctx, "About", NK_TEXT_CENTERED);

		//nk_layout_row_push(ctx, 120);
		//nk_label(ctx, "View", NK_TEXT_CENTERED);

		//nk_layout_row_push(ctx, 120);
		//nk_label(ctx, "View", NK_TEXT_CENTERED);
	}
	nk_end(ctx);
}


void UserInterface::constructSidebarSelectionTab(int *contentModeTarget, float xPos, float width) {

	if (nk_begin(ctx, ("#Control Panel Selection" + to_string(xPos)).c_str(), nk_rect(xPos, vars->toolbarHeight, width, selectionTabHeight),
				 NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR)) {
		nk_layout_row_dynamic(ctx, 15, 4);
		if (nk_button_label(ctx, "LBM")) {
			(*contentModeTarget) = 0;
		}
		if (nk_button_label(ctx, "Shadows")) {
			(*contentModeTarget) = 1;
		}
		if (nk_button_label(ctx, "Terrain")) {
			(*contentModeTarget) = 2;
		}
		if (nk_button_label(ctx, "Sky")) {
			(*contentModeTarget) = 3;
		}
		if (nk_button_label(ctx, "Cloud Vis")) {
			(*contentModeTarget) = 4;
		}
		if (nk_button_label(ctx, "Diagram Controls")) {
			(*contentModeTarget) = 5;
		}
		if (nk_button_label(ctx, "LBM DEVELOPER")) {
			(*contentModeTarget) = 6;
		}
		if (nk_button_label(ctx, "Hierarchy")) {
			(*contentModeTarget) = 7;
		}
		if (nk_button_label(ctx, "Emitters")) {
			(*contentModeTarget) = 8;
		}
		if (nk_button_label(ctx, "General Debug")) {
			(*contentModeTarget) = GENERAL_DEBUG;
		}
		if (nk_button_label(ctx, "Properties")) {
			(*contentModeTarget) = PROPERTIES;
		}

	}
	nk_end(ctx);
}

void UserInterface::constructSelectedContent(int contentMode) {

	switch (contentMode) {
		case LBM:
			constructLBMTab();
			break;
		case LIGHTING:
			constructLightingTab();
			break;
		case TERRAIN:
			constructTerrainTab();
			break;
		case SKY:
			constructSkyTab();
			break;
		case CLOUD_VIS:
			constructCloudVisualizationTab();
			break;
		case DIAGRAM:
			constructDiagramControlsTab();
			break;
		case LBM_DEBUG:
			constructLBMDebugTab();
			break;
		case SCENE_HIERARCHY:
			constructSceneHierarchyTab();
			break;
		case EMITTERS:
			constructEmittersTab();
			break;
		case GENERAL_DEBUG:
			constructGeneralDebugTab();
			break;
		case PROPERTIES:
		default:
			constructPropertiesTab();
			break;
	}

	/*
	if (contentMode == 0) {
		constructLBMTab();
	} else if (contentMode == 1) {
		constructLightingTab();
	} else if (contentMode == 2) {
		constructTerrainTab();
	} else if (contentMode == 3) {
		constructSkyTab();
	} else if (contentMode == 4) {
		constructCloudVisualizationTab();
	} else if (contentMode == 5) {
		constructDiagramControlsTab();
	} else if (contentMode == 6) {
		constructLBMDebugTab();
	} else if (contentMode == 7) {
		constructSceneHierarchyTab();
	} else if (contentMode == 8) {
		constructEmittersTab();
	} else if (contentMode == 9) {
		constructGeneralDebugTab();
	}
	*/
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

	constructTauProperty();


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


	nk_layout_row_dynamic(ctx, 15, 1);
	nk_label(ctx, "Camera movement speed", NK_TEXT_LEFT);
	nk_slider_float(ctx, 1.0f, &camera->movementSpeed, 10000.0f, 1.0f);


	// TODO - get this back in working order (and nicer looking)
	nk_layout_row_dynamic(ctx, 15, 2);
	if (nk_option_label(ctx, "Orthographic", vars->projectionMode == ORTHOGRAPHIC)) {
		vars->projectionMode = ORTHOGRAPHIC;
	}
	if (nk_option_label(ctx, "Perspective", vars->projectionMode == PERSPECTIVE)) {
		vars->projectionMode = PERSPECTIVE;
	}
	if (vars->projectionMode == PERSPECTIVE) {
		nk_slider_float(ctx, 30.0f, &vars->fov, 120.0f, 1.0f);
	}


	nk_checkbox_label(ctx, "Use freeroam camera", &vars->useFreeRoamCamera);

	constructWalkingPanel();





	//if (useFreeRoamCameraPrev != vars->useFreeRoamCamera) {
	//	if (mode >= 2) {
	//		camera = (vars->useFreeRoamCamera) ? freeRoamCamera : viewportCamera;
	//		if (vars->useFreeRoamCamera) {
	//			cout << "using freeRoamCamera from now on" << endl;
	//		}
	//	}
	//}
	//nk_colorf()


	// TODO - make this into a function
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


	constructDirLightPositionPanel();


	nk_layout_row_dynamic(ctx, 15, 1);
	nk_property_float(ctx, "focus x:", -100000.0f, &dirLight->focusPoint.x, 100000.0f, 100.0f, 100.0f);
	nk_property_float(ctx, "focus y:", -100000.0f, &dirLight->focusPoint.y, 100000.0f, 100.0f, 100.0f);
	nk_property_float(ctx, "focus z:", -100000.0f, &dirLight->focusPoint.z, 100000.0f, 100.0f, 100.0f);


	nk_layout_row_dynamic(ctx, 15, 1);
	nk_property_float(ctx, "left:", -100000.0f, &dirLight->pLeft, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "right:", -100000.0f, &dirLight->pRight, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "bottom:", -100000.0f, &dirLight->pBottom, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "top:", -100000.0f, &dirLight->pTop, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "near:", 0.1f, &dirLight->pNear, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "far:", 1.0f, &dirLight->pFar, 1000000.0f, 1000.0f, 1000.0f);


	nk_label(ctx, "EVSM", NK_TEXT_CENTERED);

	nk_checkbox_label(ctx, "use blur pass:", (int *)&evsm->useBlurPass);

	nk_property_float(ctx, "shadowBias:", 0.0f, &evsm->shadowBias, 1.0f, 0.0001f, 0.0001f);
	nk_property_float(ctx, "light bleed reduction:", 0.0f, &evsm->lightBleedReduction, 1.0f, 0.0001f, 0.0001f);
	//nk_property_float(ctx, "variance min limit:", 0.0f, &evsm.varianceMinLimit, 1.0f, 0.0001f, 0.0001f);
	nk_property_float(ctx, "exponent:", 1.0f, &evsm->exponent, 42.0f, 0.1f, 0.1f);

	nk_checkbox_label(ctx, "shadow only", &evsm->shadowOnly);
	nk_property_float(ctx, "Shadow Intensity", 0.0f, &evsm->shadowIntensity, 1.0f, 0.01f, 0.01f);


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

	if (vars->fogMode == eFogMode::LINEAR) {
		nk_property_float(ctx, "Fog min distance", 0.0f, &vars->fogMinDistance, 100000.0f, 1.0f, 10.0f);
		nk_property_float(ctx, "Fog max distance", 0.0f, &vars->fogMaxDistance, 100000.0f, 10.0f, 100.0f);
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




}

void UserInterface::constructTerrainTab() {



	nk_layout_row_dynamic(ctx, 30, 1);

	nk_label(ctx, "Terrain Controls", NK_TEXT_CENTERED);

	nk_layout_row_dynamic(ctx, 15, 1);

	HeightMap *hm = vars->heightMap;

	if (nk_combo_begin_label(ctx, tryGetTextureFilename(hm->materialMap).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
		nk_layout_row_dynamic(ctx, 15, 1);
		for (const auto& kv : *textures) {
			if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_CENTERED)) {
				vars->heightMap->materialMap = kv.second;
				nk_combo_close(ctx);
			}
		}
		nk_combo_end(ctx);
	}

	if (nk_button_label(ctx, "Terrain Generator")) {
		terrainGeneratorPopupOpened = true;
	}



	nk_checkbox_label(ctx, "Visualize normals", &vars->visualizeTerrainNormals);
	nk_property_float(ctx, "Global nrm mixing ratio:", 0.0f, &hm->globalNormalMapMixingRatio, 1.0f, 0.01f, 0.01f);
	nk_property_float(ctx, "Global nrm tiling:", 1.0f, &hm->globalNormalMapTiling, 1000.0f, 0.1f, 0.1f);
	nk_checkbox_label(ctx, "Visualize texture", &vars->heightMap->visualizeTextureMode);

	nk_checkbox_label(ctx, "Use grunge map", &hm->useGrungeMap);
	nk_property_float(ctx, "Grunge map min", 0.0f, &hm->grungeMapMin, 1.0f, 0.01f, 0.01f);
	nk_property_float(ctx, "Grunge map tiling", 1.0f, &hm->grungeMapTiling, 1000.0f, 0.1f, 0.1f);

	if (nk_combo_begin_label(ctx, tryGetTextureFilename(vars->heightMap->visTexture).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
		nk_layout_row_dynamic(ctx, 15, 1);
		if (nk_combo_item_label(ctx, "NONE", NK_TEXT_CENTERED)) {
			vars->heightMap->visTexture = nullptr;
		}
		for (const auto& kv : *textures) {
			if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_CENTERED)) {
				vars->heightMap->visTexture = kv.second;
			}
		}
		nk_combo_end(ctx);
	}



	if (vars->terrainUsesPBR) {

		for (int i = 0; i < vars->heightMap->activeMaterialCount; i++) {

			if (nk_tree_push_id(ctx, NK_TREE_NODE, ("Material " + to_string(i)).c_str(), NK_MAXIMIZED, i)) {

				nk_layout_row_dynamic(ctx, 15, 2);

				nk_label(ctx, "Albedo", NK_TEXT_LEFT);
				constructTextureSelection(&vars->heightMap->pbrMaterials[i].albedo);

				nk_label(ctx, "MetallicSmoothness", NK_TEXT_LEFT);
				constructTextureSelection(&vars->heightMap->pbrMaterials[i].metallicRoughness);

				nk_label(ctx, "Normal Map", NK_TEXT_LEFT);
				constructTextureSelection(&vars->heightMap->pbrMaterials[i].normalMap);

				nk_label(ctx, "Ambient Occlusion", NK_TEXT_LEFT);
				constructTextureSelection(&vars->heightMap->pbrMaterials[i].ao);

				nk_layout_row_dynamic(ctx, 15, 1);
				nk_property_float(ctx, "#tiling", 0.1f, &vars->heightMap->pbrMaterials[i].textureTiling, 100000.0f, 0.1f, 1.0f);


				nk_tree_pop(ctx);
			}
		}

	} else {

		nk_property_float(ctx, "Ambient intensity", 0.0f, &hm->ambientIntensity, 1.0f, 0.01f, 0.01f);
		nk_property_float(ctx, "Diffuse intensity", 0.0f, &hm->diffuseIntensity, 1.0f, 0.01f, 0.01f);
		nk_property_float(ctx, "Specular intensity", 0.0f, &hm->specularIntensity, 1.0f, 0.01f, 0.01f);
		//nk_property_float(ctx, "Diffuse intensity", 0.0f, &hm->diffuseIntensity, 1.0f - hm->specularIntensity, 0.01f, 0.01f);
		//nk_property_float(ctx, "Specular intensity", 0.0f, &hm->specularIntensity, 1.0f - hm->diffuseIntensity, 0.01f, 0.01f);


		for (int i = 0; i < vars->heightMap->activeMaterialCount; i++) {

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
				nk_property_float(ctx, "#tiling", 0.1f, &vars->heightMap->materials[i].textureTiling, 100000.0f, 0.1f, 1.0f);


				nk_tree_pop(ctx);
			}

		}
	}


	
	if (nk_button_label(ctx, "Refresh LBM HEIGHTMAP")) {
		lbm->refreshHeightMap();
	}
	nk_property_int(ctx, "x offset", 0, &vars->terrainXOffset, 1000, 1, 1);
	nk_property_int(ctx, "z offset", 0, &vars->terrainZOffset, 1000, 1, 1);


	nk_checkbox_label(ctx, "normals only", &vars->heightMap->showNormalsOnly);
	nk_property_int(ctx, "normals mode", 0, &vars->heightMap->normalsShaderMode, 10, 1, 1);


}

void UserInterface::constructTerrainGeneratorWindow() {
	if (terrainGeneratorPopupOpened) {
		float w = 500.0f;
		float h = 500.0f;
		HeightMap *hm = vars->heightMap;
		if (nk_begin(ctx, "Terrain Generator", nk_rect((vars->screenWidth - w) / 2.0f, (vars->screenHeight - h) / 2.0f, w, h), NK_WINDOW_CLOSABLE | NK_WINDOW_BORDER | NK_WINDOW_DYNAMIC | NK_WINDOW_NO_SCROLLBAR)) {

			nk_layout_row_dynamic(ctx, 15, 1);

			nk_property_float(ctx, "Minimum Height", -1000000.0f, &hm->terrainHeightRange.x, hm->terrainHeightRange.y, 100.0f, 100.0f);
			nk_property_float(ctx, "Maximum Height", hm->terrainHeightRange.x, &hm->terrainHeightRange.y, 1000000.0f, 100.0f, 100.0f);


			if (nk_combo_begin_label(ctx, hm->getDataGenerationModeString(), nk_vec2(nk_widget_width(ctx), 200))) {
				nk_layout_row_dynamic(ctx, 15, 1);
				for (int i = 0; i < HeightMap::eDataGenerationMode::_NUM_MODES; i++) {
					if (nk_combo_item_label(ctx, hm->getDataGenerationModeString(i), NK_TEXT_CENTERED)) {
						hm->dataGenerationMode = i;
						nk_combo_close(ctx);
					}
				}

				nk_combo_end(ctx);
			}

			if (hm->dataGenerationMode == HeightMap::eDataGenerationMode::HEIGHT_MAP) {
				if (nk_combo_begin_label(ctx, vars->heightMap->heightMapFilename.c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
					nk_layout_row_dynamic(ctx, 15, 1);
					for (int i = 0; i < vars->sceneFilenames.size(); i++) {
						if (nk_combo_item_label(ctx, vars->sceneFilenames[i].c_str(), NK_TEXT_CENTERED)) {
							vars->heightMap->heightMapFilename = vars->sceneFilenames[i];
							nk_combo_close(ctx);
						}
					}
					nk_combo_end(ctx);
				}
			} else if (hm->dataGenerationMode == HeightMap::eDataGenerationMode::RANDOM_PERLIN) {
				hm->constructPerlinGeneratorUITab(ctx);
			}

			if (nk_button_label(ctx, "Generate")) {
				vars->heightMap->loadAndUpload();
				lbm->refreshHeightMap();
			}
			if (nk_button_label(ctx, "Generate & Close")) {
				vars->heightMap->loadAndUpload();
				lbm->refreshHeightMap();
				terrainGeneratorPopupOpened = false;
			}
			if (nk_button_label(ctx, "Close")) {
				terrainGeneratorPopupOpened = false;
			}

			nk_end(ctx);
		} else {
			terrainGeneratorPopupOpened = false;
		}
	}
}








void UserInterface::constructSkyTab() {

	nk_layout_row_dynamic(ctx, 30, 1);

	nk_label(ctx, "Sky", NK_TEXT_CENTERED);


	constructDirLightPositionPanel();

	nk_layout_row_dynamic(ctx, 15, 1);


	nk_checkbox_label(ctx, "Skybox", &vars->drawSkybox);
	nk_checkbox_label(ctx, "Hosek", &vars->hosekSkybox);


	nk_property_double(ctx, "Turbidity", 1.0, &hosek->turbidity, 10.0, 0.1, 0.1);
	nk_property_double(ctx, "Albedo", 0.0, &hosek->albedo, 1.0, 0.01, 0.01);


	nk_property_double(ctx, "Horizon Offset", 0.001, &hosek->horizonOffset, 10.0, 0.001, 0.001);
	nk_property_float(ctx, "Sun Intensity", 0.1f, &hosek->sunIntensity, 10.0f, 0.1f, 0.1f);
	nk_property_int(ctx, "Sun Exponent", 1, &hosek->sunExponent, 1024, 1, 1);


	nk_checkbox_label(ctx, "Recalculate Live", &hosek->liveRecalc);


	if (nk_combo_begin_label(ctx, hosek->getCalcParamModeName().c_str(), nk_vec2(nk_widget_width(ctx), 60))) {
		nk_layout_row_dynamic(ctx, 15, 1);
		for (int i = 0; i < 2; i++) {
			if (nk_combo_item_label(ctx, hosek->getCalcParamModeName(i).c_str(), NK_TEXT_CENTERED)) {
				hosek->calcParamMode = i;
			}
		}
		nk_combo_end(ctx);
	}
	nk_checkbox_label(ctx, "Use Anderson's RGB normalization", &hosek->useAndersonsRGBNormalization);



	if (!hosek->liveRecalc) {
		if (nk_button_label(ctx, "Recalculate Model")) {
			hosek->update();

		}
	}



	nk_value_float(ctx, "Eta", hosek->eta);
	nk_value_float(ctx, "Eta (degrees)", hosek->getElevationDegrees());

	nk_value_float(ctx, "Sun Theta", hosek->sunTheta);
	nk_value_float(ctx, "Sun Theta (degrees)", glm::degrees(hosek->sunTheta));

	nk_checkbox_label(ctx, "Simulate sun", &vars->simulateSun);
	nk_checkbox_label(ctx, "Skip night", &dirLight->skipNightTime);
	nk_property_float(ctx, "Sun speed", 0.1f, &dirLight->circularMotionSpeed, 1000.0f, 0.1f, 0.1f);
	if (nk_option_label(ctx, "y axis", dirLight->rotationAxis == DirectionalLight::Y_AXIS)) {
		dirLight->rotationAxis = DirectionalLight::Y_AXIS;
	}
	if (nk_option_label(ctx, "z axis", dirLight->rotationAxis == DirectionalLight::Z_AXIS)) {
		dirLight->rotationAxis = DirectionalLight::Z_AXIS;
	}
	nk_property_float(ctx, "rotation radius:", 0.0f, &dirLight->radius, 10000.0f, 1.0f, 1.0f);

	nk_checkbox_label(ctx, "Use sky sun color", &vars->useSkySunColor);

}








void UserInterface::constructCloudVisualizationTab() {

	nk_layout_row_dynamic(ctx, 30, 1);

	nk_label(ctx, "Cloud Visualization", NK_TEXT_CENTERED);

	constructDirLightPositionPanel();



	nk_layout_row_dynamic(ctx, 15, 1);

	int prevNumSlices = particleRenderer->numSlices;
	nk_property_int(ctx, "Num slices", 1, &particleRenderer->numSlices, particleRenderer->maxNumSlices, 1, 1);
	if (prevNumSlices != particleRenderer->numSlices) {
		particleRenderer->numDisplayedSlices = particleRenderer->numSlices;
	}

	nk_property_int(ctx, "Num displayed slices", 0, &particleRenderer->numDisplayedSlices, particleRenderer->numSlices, 1, 1);

	nk_value_int(ctx, "Batch size", particleRenderer->batchSize);

	constructFormBoxButtonPanel();


	nk_property_float(ctx, "Shadow alpha (100x)", 0.01f, &particleRenderer->shadowAlpha100x, 100.0f, 0.01f, 0.01f);



	nk_value_vec3(ctx, particleRenderer->lightVec, "Light vector");
	nk_value_vec3(ctx, particleRenderer->viewVec, "View vector");
	nk_value_float(ctx, "Dot product", glm::dot(particleRenderer->viewVec, particleRenderer->lightVec));
	nk_property_float(ctx, "Inversion threshold", -1.0f, &particleRenderer->inversionThreshold, 1.0f, 0.01f, 0.01f);
	nk_value_bool(ctx, "Inverted rendering", particleRenderer->invertedView);
	nk_value_vec3(ctx, particleRenderer->halfVec, "Half vector");

	nk_checkbox_label(ctx, "Half vector always faces camera", &particleRenderer->forceHalfVecToFaceCam);


	nk_property_float(ctx, "Point size", 0.1f, &stlpSimCUDA->pointSize, 100000.0f, 0.1f, 0.1f);
	particleSystem->pointSize = stlpSimCUDA->pointSize;
	nk_property_float(ctx, "Opacity multiplier", 0.01f, &vars->opacityMultiplier, 10.0f, 0.01f, 0.01f);

	nk_checkbox_label(ctx, "Show particles below CCL", &particleRenderer->showParticlesBelowCCL);
	nk_checkbox_label(ctx, "use new", &particleRenderer->useVolumetricRendering);

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

	nk_property_int(ctx, "Shader set", 0, &particleRenderer->shaderSet, 2, 1, 1);
	particleRenderer->updateShaderSet();

	nk_checkbox_label(ctx, "cloud shadows", &vars->cloudsCastShadows);

	nk_property_float(ctx, "cast shadow alpha multiplier", 0.0f, &vars->cloudCastShadowAlphaMultiplier, 2.0f, 0.01f, 0.01f);


	if (nk_combo_begin_label(ctx, particleRenderer->getPhaseFunctionName(particleRenderer->phaseFunction), nk_vec2(nk_widget_width(ctx), 200))) {
		nk_layout_row_dynamic(ctx, 15, 1);

		for (int i = 0; i < ParticleRenderer::ePhaseFunction::_NUM_PHASE_FUNCTIONS; i++) {
			if (nk_combo_item_label(ctx, particleRenderer->getPhaseFunctionName(i), NK_TEXT_CENTERED)) {
				particleRenderer->phaseFunction = (ParticleRenderer::ePhaseFunction)i;
			}
		}
		nk_combo_end(ctx);
	}


	//nk_property_int(ctx, "phase function", 0, &particleRenderer->phaseFunction, 10, 1, 1);
	if (particleRenderer->phaseFunction == ParticleRenderer::ePhaseFunction::HENYEY_GREENSTEIN
		|| particleRenderer->phaseFunction == ParticleRenderer::ePhaseFunction::CORNETTE_SHANK) {

		nk_property_float(ctx, "g", -1.0f, &particleRenderer->symmetryParameter, 1.0f, 0.01f, 0.01f);

	} else if (particleRenderer->phaseFunction == ParticleRenderer::ePhaseFunction::DOUBLE_HENYEY_GREENSTEIN) {

		nk_property_float(ctx, "g1", 0.0f, &particleRenderer->symmetryParameter, 1.0f, 0.01f, 0.01f);
		nk_property_float(ctx, "g2", -1.0f, &particleRenderer->symmetryParameter2, 0.0f, 0.01f, 0.01f);
		nk_property_float(ctx, "f", 0.0f, &particleRenderer->dHenyeyGreensteinInterpolationParameter, 1.0f, 0.01f, 0.01f);

	} else if (particleRenderer->phaseFunction == ParticleRenderer::ePhaseFunction::SCHLICK) {

		nk_property_float(ctx, "k", -1.0f, &particleRenderer->symmetryParameter, 1.0f, 0.01f, 0.01f);

	}

	if (particleRenderer->phaseFunction > 0) {
		nk_checkbox_label(ctx, "Multiply by shadow intensity", &particleRenderer->multiplyPhaseByShadow);
	}


	nk_checkbox_label(ctx, "Show particle texture idx", &particleRenderer->showParticleTextureIdx);
	nk_checkbox_label(ctx, "Use atlas texture", &particleRenderer->useAtlasTexture);

	nk_checkbox_label(ctx, "Blur light texture", &particleRenderer->useBlurPass);
	nk_property_float(ctx, "blur amount", 0.0f, &particleRenderer->blurAmount, 10.0f, 0.01f, 0.01f);


}






void UserInterface::constructDiagramControlsTab() {




	nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);
	if (nk_button_label(ctx, "Recalculate Params")) {
		//lbm->resetSimulation();
		stlpDiagram->recalculateParameters();
		stlpSimCUDA->uploadDataFromDiagramToGPU();
	}


	if (nk_combo_begin_label(ctx, stlpDiagram->soundingFilename.c_str(), nk_vec2(nk_widget_width(ctx), 200.0f))) {
		if (vars->soundingDataFilenames.empty()) {
			nk_label(ctx, "empty...", NK_TEXT_LEFT);
		}

		for (int i = 0; i < vars->soundingDataFilenames.size(); i++) {
			nk_layout_row_dynamic(ctx, 15, 1);
			if (nk_combo_item_label(ctx, vars->soundingDataFilenames[i].c_str(), NK_TEXT_CENTERED)) {
				stlpDiagram->soundingFilename = vars->soundingDataFilenames[i];
			}
		}
		nk_combo_end(ctx);
	}

	if (nk_button_label(ctx, "Load Sounding File")) {
		stlpDiagram->loadSoundingData();
		stlpDiagram->recalculateAll();
		stlpSimCUDA->uploadDataFromDiagramToGPU();
	}


	if (nk_tree_push(ctx, NK_TREE_TAB, "Diagram controls", NK_MINIMIZED)) {
		nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);
		stlpDiagram->constructDiagramCurvesToolbar(ctx, this);
		nk_tree_pop(ctx);
	}

	nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);


	if (nk_button_label(ctx, "Reset to default")) {
		stlpDiagram->recalculateAll();
	}

	//if (nk_button_label(ctx, "Reset simulation")) {
	//stlpSim->resetSimulation();
	//}

	//nk_slider_float(ctx, 0.01f, &stlpSim->simulationSpeedMultiplier, 1.0f, 0.01f);

	float delta_t_prev = stlpSimCUDA->delta_t;
	nk_property_float(ctx, "delta t", 0.001f, &stlpSimCUDA->delta_t, 100.0f, 0.001f, 0.001f);
	nk_property_float(ctx, "delta t (quick)", 0.1f, &stlpSimCUDA->delta_t, 100.0f, 0.1f, 0.1f);

	//if (stlpSimCUDA->delta_t != delta_t_prev) {
	//	//stlpSimCUDA->delta_t = stlpSim->delta_t;
	//	stlpSimCUDA->updateGPU_delta_t();
	//}

	nk_property_int(ctx, "number of profiles", 2, &stlpDiagram->numProfiles, 100, 1, 1.0f); // somewhere bug when only one profile -> FIX!

	nk_property_float(ctx, "profile range", -10.0f, &stlpDiagram->convectiveTempRange, 10.0f, 0.01f, 0.01f);

	//nk_property_int(ctx, "max particles", 1, &stlpSim->maxNumParticles, 100000, 1, 10.0f);

	//nk_checkbox_label(ctx, "Simulate wind", &stlpSim->simulateWind);

	//nk_checkbox_label(ctx, "use prev velocity", &stlpSim->usePrevVelocity);

	nk_checkbox_label(ctx, "Divide Previous Velocity", &vars->dividePrevVelocity);
	if (vars->dividePrevVelocity) {
		nk_property_float(ctx, "Divisor (x100)", 100.0f, &vars->prevVelocityDivisor, 1000.0f, 0.01f, 0.01f); // [1.0, 10.0]
	}

	nk_checkbox_label(ctx, "Show CCL Level", &vars->showCCLLevelLayer);
	nk_checkbox_label(ctx, "Show EL Level", &vars->showELLevelLayer);
	nk_checkbox_label(ctx, "Show Overlay Diagram", &vars->showOverlayDiagram);
	if (vars->showOverlayDiagram) {

		int tmp = stlpDiagram->overlayDiagramResolution;
		int maxDiagramWidth = (vars->screenWidth < vars->screenHeight) ? vars->screenWidth : vars->screenHeight;
		nk_slider_float(ctx, 10.0f, &stlpDiagram->overlayDiagramResolution, maxDiagramWidth, 1.0f);


		float prevX = stlpDiagram->overlayDiagramX;
		float prevY = stlpDiagram->overlayDiagramY;
		nk_property_float(ctx, "x:", 0.0f, &stlpDiagram->overlayDiagramX, vars->screenWidth - stlpDiagram->overlayDiagramResolution, 0.1f, 0.1f);
		nk_property_float(ctx, "y:", 0.0f, &stlpDiagram->overlayDiagramY, vars->screenHeight - stlpDiagram->overlayDiagramResolution, 0.1f, 0.1f);
		
		if (tmp != stlpDiagram->overlayDiagramResolution ||
			prevX != stlpDiagram->overlayDiagramX ||
			prevY != stlpDiagram->overlayDiagramY) {
			stlpDiagram->refreshOverlayDiagram(vars->screenWidth, vars->screenHeight);
		}

		nk_checkbox_label(ctx, "Show Particles in Diagram", &vars->drawOverlayDiagramParticles);
		if (vars->drawOverlayDiagramParticles) {

		}
	}

	//nk_checkbox_label(ctx, "Use CUDA", &vars->stlpUseCUDA);

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


	struct nk_colorf tintColor;
	tintColor.r = vars->tintColor.x;
	tintColor.g = vars->tintColor.y;
	tintColor.b = vars->tintColor.z;

	nk_property_color_rgb(ctx, vars->tintColor);
	/*
	if (nk_combo_begin_color(ctx, nk_rgb_cf(tintColor), nk_vec2(nk_widget_width(ctx), 400))) {
		nk_layout_row_dynamic(ctx, 120, 1);
		tintColor = nk_color_picker(ctx, tintColor, NK_RGB);
		nk_layout_row_dynamic(ctx, 10, 1);
		tintColor.r = nk_propertyf(ctx, "#R:", 0, tintColor.r, 1.0f, 0.01f, 0.005f);
		tintColor.g = nk_propertyf(ctx, "#G:", 0, tintColor.g, 1.0f, 0.01f, 0.005f);
		tintColor.b = nk_propertyf(ctx, "#B:", 0, tintColor.b, 1.0f, 0.01f, 0.005f);
		tintColor.a = nk_propertyf(ctx, "#A:", 0, tintColor.a, 1.0f, 0.01f, 0.005f);
		vars->tintColor = glm::vec3(tintColor.r, tintColor.g, tintColor.b);
		nk_combo_end(ctx);
	}
	*/



	//for (int i = 0; i < particleSystem->emitters.size(); i++) {
	//	if (nk_tree_push_id(ctx, NK_TREE_NODE, ("#Emitter " + to_string(i)).c_str(), NK_MINIMIZED, i)) {
	//		Emitter *e = particleSystem->emitters[i];

	//		nk_layout_row_static(ctx, 15, 200, 1);
	//		nk_checkbox_label(ctx, "#enabled", &e->enabled);
	//		nk_checkbox_label(ctx, "#visible", &e->visible);

	//		nk_property_int(ctx, "#emit per step", 0, &e->numParticlesToEmitPerStep, 10000, 10, 10);

	//		PositionalEmitter *pe = dynamic_cast<PositionalEmitter *>(e);
	//		if (pe) {

	//			nk_checkbox_label(ctx, "#wiggle", &pe->wiggle);
	//			nk_property_float(ctx, "#x wiggle", 0.1f, &pe->xWiggleRange, 10.0f, 0.1f, 0.1f);
	//			nk_property_float(ctx, "#z wiggle", 0.1f, &pe->zWiggleRange, 10.0f, 0.1f, 0.1f);

	//			nk_property_float(ctx, "#x", -1000.0f, &pe->position.x, 1000.0f, 1.0f, 1.0f);
	//			//nk_property_float(ctx, "#y", -1000.0f, &pe->position.y, 1000.0f, 1.0f, 1.0f);
	//			nk_property_float(ctx, "#z", -1000.0f, &pe->position.z, 1000.0f, 1.0f, 1.0f);


	//			CircleEmitter *ce = dynamic_cast<CircleEmitter *>(pe);
	//			if (ce) {
	//				nk_property_float(ctx, "#radius", 1.0f, &ce->radius, 1000.0f, 1.0f, 1.0f);
	//			}
	//		}

	//		nk_tree_pop(ctx);
	//		//particleSystem->emitters[i]
	//	}
	//}
	nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);
	if (nk_button_label(ctx, "Activate All Particles")) {
		particleSystem->activateAllParticles();
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

		if (nk_button_label(ctx, "snap (corners by min) to ground")) {
			lbm->snapToGround();
		}

		if (nk_button_label(ctx, "SAVE CHANGES")) {

			//lbm->saveChanges(); // testing 
			lbm->stopEditing(true);
		}
		if (nk_button_label(ctx, "CANCEL")) {
			lbm->stopEditing(false);
		}

	}

	nk_layout_row_dynamic(ctx, 30.0f, 1);
	nk_label(ctx, "Streamlines", NK_TEXT_CENTERED);

	//nk_layout_row_dynamic(ctx, 200, 1); // wrapping row

	//if (nk_group_begin(ctx, "Streamlines", NK_WINDOW_BORDER)) {

	nk_layout_row_dynamic(ctx, 15, 1);
	constructTauProperty();


	if (!sps->initialized && !streamlineInitMode) {
		if (nk_button_label(ctx, "Use streamlines")) {
			//sps->init();
			streamlineInitMode = true;
		}

	} else if (sps->initialized) {

		nk_checkbox_label(ctx, "visible", &sps->visible);


		nk_layout_row(ctx, NK_STATIC, 15, 2, leftSidebarEditButtonRatio);

		if (nk_button_label(ctx, "set horizontal line")) {
			sps->setPositionInHorizontalLine();
		}
		struct nk_vec2 wpos = nk_widget_position(ctx); // this needs to be before the widget!
		wpos.x += nk_widget_size(ctx).x;
		wpos.y -= nk_widget_size(ctx).y;

		if (nk_button_image(ctx, nkEditIcon)) {
			sps->editingHorizontalParameters = true;
		}
		if (sps->editingHorizontalParameters) {
			if (nk_popup_begin(ctx, NK_POPUP_STATIC, "Horizontal Line Params Popup", 0, nk_rect(wpos.x, wpos.y, 200, 200))) {
				nk_layout_row_dynamic(ctx, 15, 1);
				nk_label(ctx, "Horizontal Line Params", NK_TEXT_LEFT);

				nk_property_float(ctx, "xOffset (ratio)", 0.0f, &sps->hParams.xOffset, 0.99f, 0.01f, 0.01f);
				nk_property_float(ctx, "yOffset (ratio)", 0.0f, &sps->hParams.yOffset, 0.99f, 0.01f, 0.01f);

				if (nk_button_label(ctx, "Save & Apply")) {
					sps->setPositionInHorizontalLine();
					sps->editingHorizontalParameters = false;
					nk_popup_close(ctx);
				}
				if (nk_button_label(ctx, "Save")) {
					sps->editingHorizontalParameters = false;
					nk_popup_close(ctx);
				}
				nk_popup_end(ctx);
			} else {
				sps->editingHorizontalParameters = false;
			}
		}

		if (nk_button_label(ctx, "set vertical line")) {
			sps->setPositionInVerticalLine();
		}
		wpos = nk_widget_position(ctx); // this needs to be before the widget!
		wpos.x += nk_widget_size(ctx).x;
		wpos.y -= nk_widget_size(ctx).y;
		if (nk_button_image(ctx, nkEditIcon)) {
			sps->editingVerticalParameters = true;
		}
		if (sps->editingVerticalParameters) {
			if (nk_popup_begin(ctx, NK_POPUP_STATIC, "Vertical Line Params Popup", 0, nk_rect(wpos.x, wpos.y, 200, 200))) {
				nk_layout_row_dynamic(ctx, 15, 1);
				nk_label(ctx, "Vertical Line Params", NK_TEXT_LEFT);

				nk_property_float(ctx, "xOffset (ratio)", 0.0f, &sps->vParams.xOffset, 0.99f, 0.01f, 0.01f);
				nk_property_float(ctx, "zOffset (ratio)", 0.0f, &sps->vParams.zOffset, 0.99f, 0.01f, 0.01f);

				if (nk_button_label(ctx, "Save & Apply")) {
					sps->setPositionInVerticalLine();
					sps->editingVerticalParameters = false;
					nk_popup_close(ctx);
				}
				if (nk_button_label(ctx, "Save")) {
					sps->editingVerticalParameters = false;
					nk_popup_close(ctx);
				}
				nk_popup_end(ctx);
			} else {
				sps->editingVerticalParameters = false;
			}
		}
		nk_layout_row_dynamic(ctx, 15.0f, 1);


		if (sps->active) {
			if (nk_button_label(ctx, "Deactivate streamlines")) {
				sps->deactivate();
			}
		} else {
			if (nk_button_label(ctx, "Activate streamlines")) {
				sps->activate();
			}
		}

		if (nk_button_label(ctx, "Reset")) {
			sps->reset();
		}


		nk_checkbox_label(ctx, "live line cleanup (DEBUG)", &sps->liveLineCleanup);

		struct nk_rect bounds;

		bounds = nk_widget_bounds(ctx);
		if (nk_button_label(ctx, "Teardown")) {
			sps->tearDown();
		}
		if (nk_input_is_mouse_hovering_rect(ctx_in, bounds)) {
			nk_layout_row_dynamic(ctx, 15, 1);
			nk_tooltip(ctx, "Teardown current buffers - user can then create new streamline environment.");
		}


	} else if (streamlineInitMode) {

		nk_property_int(ctx, "max streamlines", 1, &sps->maxNumStreamlines, 10000, 1, 1);
		nk_property_int(ctx, "max streamline length", 1, &sps->maxStreamlineLength, 1000, 1, 1);
		//nk_property_int(ctx, "streamline sampling", 1, &sps->sampling)

		if (nk_button_label(ctx, "Apply settings")) {
			cout << "Initializing streamline data..." << endl;
			sps->init();
			streamlineInitMode = false;
		}

	}
	//nk_group_end(ctx);

//}



}

void UserInterface::constructSceneHierarchyTab() {


	hierarchyIdCounter = 0;
	activeActors.clear();

	nk_layout_row_dynamic(ctx, 30, 1);
	nk_label(ctx, "Scene Hierarchy", NK_TEXT_CENTERED);


	//addSceneHierarchyActor(scene->root);

	for (int i = 0; i < scene->root->children.size(); i++) {
		addSceneHierarchyActor(scene->root->children[i]);
	}


}

void UserInterface::addSceneHierarchyActor(Actor * actor) {
	if (actor == nullptr) {
		return;
	}
	hierarchyIdCounter++;

	nk_layout_row_dynamic(ctx, 15, 1);


	if (actor->children.size() > 0) {


		//if (nk_tree_element_push_id(ctx, NK_TREE_NODE, actor->name.c_str(), NK_MINIMIZED, 0, hierarchyIdCounter)) {
		if (nk_tree_element_push_id(ctx, NK_TREE_NODE, actor->name.c_str(), NK_MINIMIZED, &actor->selected, hierarchyIdCounter)) {


			//nk_layout_row_dynamic(ctx, 250, 1);
			//if (nk_group_begin_titled(ctx, to_string(hierarchyIdCounter).c_str(), "Transform", NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR)) {
			//	nk_layout_row_dynamic(ctx, 15, 1);
			//	nk_property_vec3(actor->transform.position, -1000000.0f, 1000000.0f, 0.1f, 10.0f, "pos");
			//	nk_property_vec3(actor->transform.rotation, 0.0f, 360.0f, 0.1f, 0.1f, "rot");
			//	nk_property_vec3(actor->transform.scale, 0.0f, 1000.0f, 0.1f, 0.1f, "scale");
			//	//nk_property_float(ctx, "scale", 0.0f, &actor->transform.scale.x, 10000.0f, 0.1f, 0.1f);
			//	//// quick hack
			//	//actor->transform.scale.y = actor->transform.scale.x;
			//	//actor->transform.scale.z = actor->transform.scale.x;

			//	nk_group_end(ctx);
			//}

			for (int i = 0; i < actor->children.size(); i++) {
				addSceneHierarchyActor(actor->children[i]);
			}

			nk_tree_pop(ctx);
		}
	} else {
		struct nk_rect w;
		w = nk_layout_widget_bounds(ctx);

		//nk_layout_row_dynamic(ctx, 15, 1);
		nk_layout_row_begin(ctx, NK_STATIC, 15.0f, 2);
		nk_layout_row_push(ctx, w.w - 15.0f);
		nk_selectable_label(ctx, actor->name.c_str(), NK_TEXT_LEFT, &actor->selected);
		nk_layout_row_push(ctx, 15.0f);


		if (nk_button_symbol(ctx, actor->visible ? NK_SYMBOL_CIRCLE_SOLID : NK_SYMBOL_CIRCLE_OUTLINE)) {
			actor->visible = !actor->visible;
		}
		nk_layout_row_end(ctx);

	}
	if (actor->selected) {
		activeActors.push_back(actor);
	}
}


void UserInterface::constructEmittersTab() {

	vector<int> emitterIndicesToDelete;

	for (int i = 0; i < particleSystem->emitters.size(); i++) {
		if (nk_tree_push_id(ctx, NK_TREE_NODE, ("#Emitter " + to_string(i)).c_str(), NK_MINIMIZED, i)) {
			Emitter *e = particleSystem->emitters[i];

			nk_layout_row_static(ctx, 15, 200, 1);
			nk_checkbox_label(ctx, "#enabled", &e->enabled);
			nk_checkbox_label(ctx, "#visible", &e->visible);

			nk_property_int(ctx, "#emit per step", 0, &e->numParticlesToEmitPerStep, 10000, 10, 10);

			nk_property_int(ctx, "min profile index", 0, &e->minProfileIndex, e->maxProfileIndex, 1, 1);
			nk_property_int(ctx, "max profile index", e->minProfileIndex, &e->maxProfileIndex, stlpDiagram->numProfiles - 1, 1, 1);

			PositionalEmitter *pe = dynamic_cast<PositionalEmitter *>(e);
			if (pe) {

				nk_checkbox_label(ctx, "#wiggle", &pe->wiggle);
				nk_property_float(ctx, "#x wiggle", 0.1f, &pe->xWiggleRange, 10.0f, 0.1f, 0.1f);
				nk_property_float(ctx, "#z wiggle", 0.1f, &pe->zWiggleRange, 10.0f, 0.1f, 0.1f);

				nk_property_float(ctx, "#x", -1000.0f, &pe->position.x, 1000.0f, 1.0f, 1.0f);
				//nk_property_float(ctx, "#y", -1000.0f, &pe->position.y, 1000.0f, 1.0f, 1.0f);
				nk_property_float(ctx, "#z", -1000.0f, &pe->position.z, 1000.0f, 1.0f, 1.0f);


				CircleEmitter *ce = dynamic_cast<CircleEmitter *>(pe);
				if (ce) {
					nk_property_float(ctx, "#radius", 1.0f, &ce->radius, 1000.0f, 1.0f, 1.0f);
				}



			}

			if (nk_button_label(ctx, "Delete emitter")) {
				//particleSystem->deleteEmitter(i);
				emitterIndicesToDelete.push_back(i); // do not delete when iterating through the vector
			}

			nk_tree_pop(ctx);
			//particleSystem->emitters[i]
		}
	}

	for (const auto &i : emitterIndicesToDelete) {
		particleSystem->deleteEmitter(i);
	}


	nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);
	if (nk_button_label(ctx, "Activate All Particles")) {
		particleSystem->activateAllParticles();
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


}

void UserInterface::constructGeneralDebugTab() {
	stringstream ss;

	nk_layout_row_static(ctx, 15, vars->rightSidebarWidth, 1);

	/*
	ss.clear();
	ss << "Delta time: " << fixed << setprecision(2) << (deltaTime * 1000.0);

	//string fpsStr = "delta time: " + to_string(deltaTime * 1000.0);
	nk_label(ctx, ss.str().c_str(), NK_TEXT_CENTERED);
	*/
	stringstream().swap(ss);
	ss << "Delta time: " << fixed << setprecision(4) << prevAvgDeltaTime << " [ms] (" << setprecision(0) << prevAvgFPS << " FPS)";
	nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);

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
	nk_value_int(ctx, "terrain width resolution", vars->heightMap->width);
	nk_value_int(ctx, "terrain depth resolution", vars->heightMap->height);




	nk_property_vec3(ctx, 0.0f, dirLight->color, 100.0f, 1.0f, 1.0f, "dir light color", eVecNaming::COLOR);
	nk_property_float(ctx, "dir light intensity", 0.0f, &dirLight->intensity, 1000.0f, 0.01f, 0.01f);


	if (nk_button_label(ctx, "Reload shaders (recompile all) - EXPERIMENTAL")) {
		ShaderManager::loadShaders();
	}

}

void UserInterface::constructPropertiesTab() {

	if (activeActors.empty()) {
		nk_layout_row_dynamic(ctx, 60, 1);
		nk_label_wrap(ctx, "Select objects in hierarchy to display their properties here...");
	}

	for (const auto &actor : activeActors) {

		nk_layout_row_dynamic(ctx, 15, 1);
		nk_label(ctx, actor->name.c_str(), NK_TEXT_LEFT);
		nk_checkbox_label(ctx, "visible", &actor->visible);

		if (!actor->isRootChild()) {
			if (nk_button_label(ctx, "move up a level (unparent)")) {
				actor->unparent();
			}
		}
		if (nk_button_label(ctx, "Snap to Ground")) {
			actor->snapToGround(vars->heightMap);
		}


		nk_layout_row_dynamic(ctx, 250, 1);
		if (nk_group_begin_titled(ctx, to_string(hierarchyIdCounter).c_str(), "Transform", NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR)) {
			nk_layout_row_dynamic(ctx, 15, 1);
			nk_property_vec3(ctx, -1000000.0f, actor->transform.position, 1000000.0f, 0.01f, 0.1f, "pos");
			nk_property_vec3(ctx, 0.0f, actor->transform.rotation, 360.0f, 0.1f, 0.1f, "rot");
			nk_property_vec3(ctx, 0.0f, actor->transform.scale, 1000.0f, 0.1f, 0.1f, "scale");
			//nk_property_float(ctx, "scale", 0.0f, &actor->transform.scale.x, 10000.0f, 0.1f, 0.1f);
			//// quick hack
			//actor->transform.scale.y = actor->transform.scale.x;
			//actor->transform.scale.z = actor->transform.scale.x;

			nk_group_end(ctx);
		}


		Model *model = dynamic_cast<Model *>(actor);

		if (model) {

			model->constructUserInterfaceTab(ctx, vars->heightMap);


		}
		



	}



}


void UserInterface::constructDebugTab() {
}

void UserInterface::constructFavoritesMenu() {
	nk_layout_row_push(ctx, 120);
	if (nk_menu_begin_label(ctx, "Favorites", NK_TEXT_CENTERED, nk_vec2(300, 250))) {
		nk_layout_row_dynamic(ctx, 15, 1);

		constructTauProperty();
		if (nk_button_label(ctx, "Form BOX")) {
			particleSystem->formBox();
		}

		nk_checkbox_label(ctx, "Apply LBM", &vars->applyLBM);
		nk_checkbox_label(ctx, "Apply STLP", &vars->applySTLP);
		nk_property_float(ctx, "delta t (quick)", 0.1f, &stlpSimCUDA->delta_t, 100.0f, 0.1f, 0.1f);

		constructWalkingPanel();

		constructDirLightPositionPanel();

		// temporary
		nk_checkbox_label(ctx, "cloud shadows", &vars->cloudsCastShadows);


		nk_menu_end(ctx);
	}
}

void UserInterface::constructDirLightPositionPanel() {
	nk_layout_row_dynamic(ctx, 15, 1);
	nk_property_vec3(ctx, -1000000.0f, dirLight->position, 1000000.0f, 100.0f, 100.0f, "Sun position");
	//nk_label(ctx, "Sun position", NK_TEXT_LEFT);
	//nk_property_float(ctx, "#x:", -1000.0f, &dirLight->position.x, 1000.0f, 1.0f, 1.0f);
	//nk_property_float(ctx, "#y:", -1000.0f, &dirLight->position.y, 1000.0f, 1.0f, 1.0f);
	//nk_property_float(ctx, "#z:", -1000.0f, &dirLight->position.z, 1000.0f, 1.0f, 1.0f);
}

void UserInterface::constructFormBoxButtonPanel() {

	//ctx->style.button.touch_padding
	float ratio_two[] = { vars->leftSidebarWidth - 20.0f,  15.0f };
	//nk_layout_row_dynamic(ctx, 15, 2);

	nk_layout_row(ctx, NK_STATIC, 15, 2, ratio_two);

	// testing
	float prevButtonRounding = ctx->style.button.rounding;
	struct nk_vec2 prevButtonPadding = ctx->style.button.padding;

	ctx->style.button.rounding = 0.0f;
	ctx->style.button.padding = nk_vec2(0.0f, 0.0f);
	if (nk_button_label(ctx, "Form BOX")) {
		particleSystem->formBox();
	}

	struct nk_vec2 wpos = nk_widget_position(ctx); // this needs to be before the widget!
	wpos.x += nk_widget_size(ctx).x;
	wpos.y -= nk_widget_size(ctx).y;


	if (nk_button_image(ctx, nkEditIcon)) {
		particleSystem->editingFormBox = true;
	}
	if (particleSystem->editingFormBox) {
		//static struct nk_rect s = { 20, 100, 200, 200 }
		if (nk_popup_begin(ctx, NK_POPUP_STATIC, "Form Box Settings", 0, nk_rect(wpos.x, wpos.y, 200, 250))) {
			nk_layout_row_dynamic(ctx, 15, 1);
			nk_label(ctx, "Form box settings", NK_TEXT_LEFT);
			nk_property_vec3(ctx, -100000.0f, particleSystem->newFormBoxSettings.position, 100000.0f, 10.0f, 10.0f, "Position");
			nk_property_vec3(ctx, 100.0f, particleSystem->newFormBoxSettings.size, 100000.0f, 10.0f, 10.0f, "Size");

			if (nk_button_label(ctx, "Save & Apply")) {
				particleSystem->formBoxSettings = particleSystem->newFormBoxSettings;
				particleSystem->formBox();
				particleSystem->editingFormBox = false;
				nk_popup_close(ctx);
			}
			if (nk_button_label(ctx, "Save")) {
				particleSystem->formBoxSettings = particleSystem->newFormBoxSettings;
				particleSystem->editingFormBox = false;
				nk_popup_close(ctx);
			}
			if (nk_button_label(ctx, "Discard")) {
				particleSystem->newFormBoxSettings = particleSystem->formBoxSettings;
				particleSystem->editingFormBox = false;
				nk_popup_close(ctx);
			}

			nk_popup_end(ctx);
		} else {
			particleSystem->editingFormBox = false;
		}
	}


	ctx->style.button.rounding = prevButtonRounding;
	ctx->style.button.padding = prevButtonPadding;

	nk_layout_row_dynamic(ctx, 15, 1);
}

void UserInterface::constructTextureSelection(Texture **targetTexturePtr) {
	if (nk_combo_begin_label(ctx, getTextureName(*targetTexturePtr).c_str(), nk_vec2(400, 200))) {
		nk_layout_row_dynamic(ctx, 15, 1);
		if (nk_combo_item_label(ctx, "NONE", NK_TEXT_CENTERED)) {
			*targetTexturePtr = nullptr;
		}
		for (const auto& kv : *textures) {
			if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_CENTERED)) {
				(*targetTexturePtr) = kv.second;
			}
		}
		nk_combo_end(ctx);
	}
}

void UserInterface::constructTauProperty() {
	nk_property_float(ctx, "Tau:", 0.5005f, &lbm->tau, 10.0f, 0.005f, 0.005f);
}

void UserInterface::constructWalkingPanel() {
	if (vars->useFreeRoamCamera) {
		FreeRoamCamera *fcam = (FreeRoamCamera*)camera;
		if (fcam) {
			int wasWalking = fcam->walking;
			nk_checkbox_label(ctx, "Walking", &fcam->walking);
			if (!wasWalking && fcam->walking) {
				fcam->snapToGround();
				fcam->movementSpeed = 1.4f;
			}

			nk_property_float(ctx, "Player Height", 0.0f, &fcam->playerHeight, 10.0f, 0.01f, 0.01f);
		}
	}
}

std::string UserInterface::tryGetTextureFilename(Texture * tex) {
	if (tex == nullptr) {
		return "NONE";
	} else {
		return tex->filename;
	}
}

