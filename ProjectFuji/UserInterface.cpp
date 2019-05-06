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
#include "EmitterBrushMode.h"

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


	leftSidebarEditButtonRatio[0] = leftSidebarWidth - 20.0f;
	leftSidebarEditButtonRatio[1] = 15.0f;


	ctx->style.button.rounding = 0.0f;
	ctx->style.button.image_padding = nk_vec2(0.0f, 0.0f);
	activeButtonStyle = ctx->style.button;

	inactiveButtonStyle = activeButtonStyle;
	inactiveButtonStyle.normal = nk_style_item_color(nk_rgb(40, 40, 40));
	inactiveButtonStyle.hover = nk_style_item_color(nk_rgb(40, 40, 40));
	inactiveButtonStyle.active = nk_style_item_color(nk_rgb(40, 40, 40));
	inactiveButtonStyle.border_color = nk_rgb(60, 60, 60);
	inactiveButtonStyle.text_background = nk_rgb(60, 60, 60);
	inactiveButtonStyle.text_normal = nk_rgb(60, 60, 60);
	inactiveButtonStyle.text_hover = nk_rgb(60, 60, 60);
	inactiveButtonStyle.text_active = nk_rgb(60, 60, 60);

	hudRect = nk_rect((float)(vars->screenWidth - vars->rightSidebarWidth - hudWidth), (float)vars->toolbarHeight, hudWidth, hudHeight);
}

UserInterface::~UserInterface() {
	nk_glfw3_shutdown();
}

void UserInterface::draw() {
	nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
}


void UserInterface::constructUserInterface() {
	if (vars->hideUI) {
		return;
	}

	nk_glfw3_new_frame();


	textures = TextureManager::getTexturesMapPtr();

	const struct nk_input *in = &ctx->input;

	//ctx->style.window.border = 5.0f;

	//ctx->style.tab.rounding = 0.0f;
	//ctx->style.button.rounding = 0.0f;
	ctx->style.property.rounding = 0.0f;

	constructLeftSidebar();
	constructRightSidebar();

	ctx->style.window.padding = popupWindowPadding;

	constructTerrainGeneratorWindow();
	constructEmitterCreationWindow();
	constructSaveParticlesWindow();
	constructLoadParticlesWindow();
	constructHUD();

	if (aboutWindowOpened) {


		if (nk_begin(ctx, "Orographic Cloud Simulator", nk_rect((float)(vars->screenWidth / 2 - 250), (float)(vars->screenHeight / 2 - 250), 500.0f, 500.0f), NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_CLOSABLE | NK_WINDOW_DYNAMIC)) {
			nk_layout_row_dynamic(ctx, 20.0f, 1);

			nk_label(ctx, "Author: Martin Cap", NK_TEXT_LEFT);
			nk_label(ctx, "Email: martincap94@gmail.com", NK_TEXT_LEFT);
			nk_label(ctx, "Website: www.martincap.io", NK_TEXT_LEFT);



		} else {
			aboutWindowOpened = false;
		}
		nk_end(ctx);
	}
	constructHorizontalBar();


}

bool UserInterface::isAnyWindowHovered() {
	return nk_window_is_any_hovered(ctx) != 0;
}

void UserInterface::nk_property_vec2(nk_context * ctx, float min, glm::vec2 & target, float max, float step, float pixStep, std::string label, bool labelIsHeader, eVecNaming namingConvention) {
	if (!label.empty()) {
		if (labelIsHeader) {
			nk_label_header(ctx, label.c_str());
		} else {
			nk_label(ctx, label.c_str(), NK_TEXT_CENTERED);
		}
	}

	int idxOffset = namingConvention * 4;
	for (int i = 0; i < 2; i++) {
		nk_property_float(ctx, vecNames[i + idxOffset], min, &target[i], max, step, pixStep);
	}
}

void UserInterface::nk_property_vec3(nk_context * ctx, float min, glm::vec3 & target, float max, float step, float pixStep, std::string label, bool labelIsHeader, eVecNaming namingConvention) {
	if (!label.empty()) {
		if (labelIsHeader) {
			nk_label_header(ctx, label.c_str());
		} else {
			nk_label(ctx, label.c_str(), NK_TEXT_CENTERED);
		}
	}

	int idxOffset = namingConvention * 4;
	for (int i = 0; i < 3; i++) {
		nk_property_float(ctx, vecNames[i + idxOffset], min, &target[i], max, step, pixStep);
	}
}

void UserInterface::nk_property_vec4(nk_context * ctx, glm::vec4 & target) {
	REPORT_NOT_IMPLEMENTED();
}

void UserInterface::nk_property_color_rgb(nk_context * ctx, glm::vec3 & target, const char * label) {
	struct nk_colorf tmp;
	tmp.r = target.r;
	tmp.g = target.g;
	tmp.b = target.b;
	nk_layout_row_begin(ctx, NK_DYNAMIC, 15.0f, 2);
	nk_layout_row_push(ctx, 0.2f);
	nk_label(ctx, label, NK_TEXT_RIGHT);
	nk_layout_row_push(ctx, 0.8f);
	if (nk_combo_begin_color(ctx, nk_rgb_cf(tmp), nk_vec2(nk_widget_width(ctx), 400))) {
		nk_layout_row_dynamic(ctx, 120, 1);
		tmp = nk_color_picker(ctx, tmp, NK_RGB);
		nk_layout_row_dynamic(ctx, 15, 1);
		target.r = nk_propertyf(ctx, "#R:", 0, tmp.r, 1.0f, 0.01f, 0.005f);
		target.g = nk_propertyf(ctx, "#G:", 0, tmp.g, 1.0f, 0.01f, 0.005f);
		target.b = nk_propertyf(ctx, "#B:", 0, tmp.b, 1.0f, 0.01f, 0.005f);
		nk_combo_end(ctx);
	}
	nk_layout_row_end(ctx);
}

void UserInterface::nk_property_color_rgba(nk_context * ctx, glm::vec4 & target, const char * label) {
	struct nk_colorf tmp;
	tmp.r = target.r;
	tmp.g = target.g;
	tmp.b = target.b;
	tmp.a = target.a;
	nk_layout_row_begin(ctx, NK_DYNAMIC, 15.0f, 2);
	nk_layout_row_push(ctx, 0.2f);
	nk_label(ctx, label, NK_TEXT_RIGHT);
	nk_layout_row_push(ctx, 0.8f);
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
	nk_layout_row_end(ctx);
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

void UserInterface::nk_label_header(nk_context * ctx, const char * headerString, bool resetRowAfter, nk_text_alignment textAlignment) {
	nk_layout_row_dynamic(ctx, headerHeight, 1);
	nk_label(ctx, headerString, textAlignment);
	if (resetRowAfter) {
		nk_layout_row_dynamic(ctx, wh, 1);
	}
}



void UserInterface::constructLeftSidebar() {
	ctx->style.window.padding = sidebarPadding;


	ctx->style.window.border = leftSidebarBorderWidth;

	ctx->style.button.padding = nk_vec2(0.0f, 0.0f);
	ctx->style.button.border = 0.0f;

	constructSidebarSelectionTab(&leftSidebarContentMode, 0, leftSidebarWidth + 20.0f);

	if (nk_begin(ctx, "Left Sidebar", nk_rect(0, vars->toolbarHeight + selectionTabHeight, leftSidebarWidth + 20.0f, vars->screenHeight - vars->toolbarHeight - selectionTabHeight),
				 NK_WINDOW_BORDER /*| NK_WINDOW_NO_SCROLLBAR*/)) {
		constructSelectedContent(leftSidebarContentMode, S_LEFT);

	}
	nk_end(ctx);


}

void UserInterface::constructRightSidebar() {
	ctx->style.window.padding = sidebarPadding;

	constructSidebarSelectionTab(&rightSidebarContentMode, (float)(vars->screenWidth - vars->rightSidebarWidth), (float)vars->rightSidebarWidth);


	if (nk_begin(ctx, "Right Sidebar", nk_rect((float)(vars->screenWidth - vars->rightSidebarWidth), (float)(vars->toolbarHeight + selectionTabHeight), (float)vars->rightSidebarWidth, /*vars->debugTabHeight*/(float)(vars->screenHeight - vars->toolbarHeight - selectionTabHeight)), NK_WINDOW_BORDER /*| NK_WINDOW_NO_SCROLLBAR*/)) {

		constructSelectedContent(rightSidebarContentMode, S_RIGHT);
		//constructGeneralDebugTab();

	}
	nk_end(ctx);
}

void UserInterface::constructHorizontalBar() {

	ctx->style.window.padding = horizontalBarPadding;

	if (nk_begin(ctx, "HorizontalBar", nk_rect(0.0f, 0.0f, (float)vars->screenWidth, (float)vars->toolbarHeight), NK_WINDOW_NO_SCROLLBAR)) {

		int numToolbarItems = 4;

		nk_menubar_begin(ctx);

		/* menu #1 */
		nk_layout_row_begin(ctx, NK_STATIC, (float)vars->toolbarHeight, 5);
		nk_layout_row_push(ctx, 120);
		if (nk_menu_begin_label(ctx, "File", NK_TEXT_CENTERED, toolbarMenuSize)) {
			nk_layout_row_dynamic(ctx, 15, 1);
			if (nk_menu_item_label(ctx, "Load Particles from File", NK_TEXT_LEFT)) {
				openPopupWindow(loadParticlesWindowOpened);
				particleSystem->loadParticleSaveFiles();
			}
			if (nk_menu_item_label(ctx, "Save Particles to File", NK_TEXT_LEFT)) {
				openPopupWindow(saveParticlesWindowOpened);
			}
			if (nk_menu_item_label(ctx, "Exit Application", NK_TEXT_LEFT)) {
				vars->appRunning = false;
			}
			nk_menu_end(ctx);
		}


		nk_layout_row_push(ctx, 120);
		if (nk_menu_begin_label(ctx, "View", NK_TEXT_CENTERED, toolbarMenuSize)) {
			nk_layout_row_dynamic(ctx, 15, 1);

			nk_checkbox_label(ctx, "Render Mode", &vars->renderMode);

			if (viewportMode == eViewportMode::VIEWPORT_3D) {
				if (nk_menu_item_label(ctx, "Diagram View (2)", NK_TEXT_LEFT)) {
					viewportMode = eViewportMode::DIAGRAM;
					glfwSwapInterval(1);
				}
			} else if (viewportMode == eViewportMode::DIAGRAM) {
				if (nk_menu_item_label(ctx, "3D Viewport (1)", NK_TEXT_LEFT)) {
					viewportMode = eViewportMode::VIEWPORT_3D;
					glfwSwapInterval(vars->vsync);

				}
			}

			if (!vars->useFreeRoamCamera) {
				nk_label(ctx, "Camera Settings", NK_TEXT_CENTERED);
				if (nk_menu_item_label(ctx, "Front View (I)", NK_TEXT_LEFT)) {
					camera->setView(Camera::VIEW_FRONT);
				}
				if (nk_menu_item_label(ctx, "Side View (O)", NK_TEXT_LEFT)) {
					camera->setView(Camera::VIEW_SIDE);
				}
				if (nk_menu_item_label(ctx, "Top View (P)", NK_TEXT_LEFT)) {
					camera->setView(Camera::VIEW_TOP);
				}
			}

			if (vars->drawSkybox) {
				if (nk_menu_item_label(ctx, "Hide Skybox", NK_TEXT_LEFT)) {
					vars->drawSkybox = false;
				}
			} else {
				if (nk_menu_item_label(ctx, "Show Skybox", NK_TEXT_LEFT)) {
					vars->drawSkybox = true;
				}
			}


			nk_menu_end(ctx);

		}
		nk_layout_row_push(ctx, 120);
		if (nk_menu_begin_label(ctx, "About", NK_TEXT_CENTERED, toolbarMenuSize)) {
			nk_layout_row_dynamic(ctx, 15.0f, 1);
			if (nk_menu_item_label(ctx, "Show About", NK_TEXT_CENTERED)) {
				openPopupWindow(aboutWindowOpened);
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
			(*contentModeTarget) = LBM;
		}
		if (nk_button_label(ctx, "Lighting")) {
			(*contentModeTarget) = LIGHTING;
		}
		if (nk_button_label(ctx, "Terrain")) {
			(*contentModeTarget) = TERRAIN;
		}
		if (nk_button_label(ctx, "Sky")) {
			(*contentModeTarget) = SKY;
		}
		if (nk_button_label(ctx, "Cloud Vis")) {
			(*contentModeTarget) = CLOUD_VIS;
		}
		if (nk_button_label(ctx, "STLP")) {
			(*contentModeTarget) = DIAGRAM;
		}
		//if (nk_button_label(ctx, "LBM DEVELOPER")) {
		//	(*contentModeTarget) = LBM_DEBUG;
		//}
		if (nk_button_label(ctx, "Emitters")) {
			(*contentModeTarget) = EMITTERS;
		}
		if (nk_button_label(ctx, "View")) {
			(*contentModeTarget) = VIEW;
		}
		if (nk_button_label(ctx, "Debug")) {
			(*contentModeTarget) = GENERAL_DEBUG;
		}
		if (nk_button_label(ctx, "Hierarchy")) {
			(*contentModeTarget) = SCENE_HIERARCHY;
		}
		if (nk_button_label(ctx, "Properties")) {
			(*contentModeTarget) = PROPERTIES;
		}
		if (nk_button_label(ctx, "Particles")) {
			(*contentModeTarget) = PARTICLE_SYSTEM;
		}

	}
	nk_end(ctx);
}

void UserInterface::constructSelectedContent(int contentMode, int side) {

	switch (contentMode) {
		case LBM:
			constructLBMTab(side);
			break;
		case LIGHTING:
			constructLightingTab(side);
			break;
		case TERRAIN:
			constructTerrainTab(side);
			break;
		case SKY:
			constructSkyTab(side);
			break;
		case CLOUD_VIS:
			constructCloudVisualizationTab(side);
			break;
		case DIAGRAM:
			constructDiagramControlsTab(side);
			break;
		case LBM_DEBUG:
			constructLBMDebugTab(side);
			break;
		case SCENE_HIERARCHY:
			constructSceneHierarchyTab(side);
			break;
		case EMITTERS:
			constructEmittersTab(side);
			break;
		case GENERAL_DEBUG:
			constructGeneralDebugTab(side);
			break;
		case VIEW:
			constructViewTab(side);
			break;
		case PARTICLE_SYSTEM:
			constructParticleSystemTab(side);
			break;
		case PROPERTIES:
		default:
			constructPropertiesTab(side);
			break;
	}
}


void UserInterface::constructLBMTab(int side) {
	nk_label_header(ctx, "LBM", false);


	nk_label_header(ctx, "LBM Controls");


	const char *buttonDescription = vars->applyLBM ? "Pause" : "Play";
	if (nk_button_label(ctx, buttonDescription)) {
		vars->applyLBM = !vars->applyLBM;
	}

	if (nk_button_label(ctx, "Reset")) {
		lbm->resetSimulation();
	}

	//if (nk_button_label(ctx, "Refresh Heightmap")) {
	//	lbm->refreshHeightMap();
	//}

	constructTauProperty();

	nk_property_int(ctx, "LBM step frame", 1, &vars->lbmStepFrame, 100, 1, 1);


	nk_label_header(ctx, "Inlet Settings");


	nk_label(ctx, "Inlet velocity:", NK_TEXT_LEFT);

	nk_property_float(ctx, "x:", -1.0f, &lbm->inletVelocity.x, 1.0f, 0.01f, 0.005f);
	nk_property_float(ctx, "y:", -1.0f, &lbm->inletVelocity.y, 1.0f, 0.01f, 0.005f);
	nk_property_float(ctx, "z:", -1.0f, &lbm->inletVelocity.z, 1.0f, 0.01f, 0.005f);


	//nk_layout_row_dynamic(ctx, 15, 1);
	nk_layout_row_begin(ctx, NK_DYNAMIC, 15.0f, 2);

	nk_layout_row_push(ctx, 0.38f);
	nk_label(ctx, "Respawn Mode:", NK_TEXT_CENTERED);

	nk_layout_row_push(ctx, 0.62f);
	if (nk_combo_begin_label(ctx, lbm->getRespawnModeString(lbm->respawnMode), nk_vec2(nk_widget_width(ctx), 300.0f))) {
		nk_layout_row_dynamic(ctx, 15, 1);
		for (int i = 0; i < LBM3D_1D_indices::eRespawnMode::_NUM_RESPAWN_MODES; i++) {
			if (nk_combo_item_label(ctx, lbm->getRespawnModeString(i), NK_TEXT_LEFT)) {
				lbm->respawnMode = i;
			}
		}
		nk_combo_end(ctx);
	}
	nk_layout_row_end(ctx);

	/*
	if (nk_option_label(ctx, "Keep Position", lbm->respawnMode == LBM3D_1D_indices::eRespawnMode::CYCLE_ALL)) {
		lbm->respawnMode = LBM3D_1D_indices::eRespawnMode::CYCLE_ALL;
	}
	if (nk_option_label(ctx, "Random (Uniform)", lbm->respawnMode == LBM3D_1D_indices::eRespawnMode::RANDOM_UNIFORM)) {
		lbm->respawnMode = LBM3D_1D_indices::eRespawnMode::RANDOM_UNIFORM;
	}
	*/

	/*
	nk_layout_row_dynamic(ctx, 15, 1);
	nk_label(ctx, "LBM Out of Bounds Mode", NK_TEXT_CENTERED);
	nk_layout_row_dynamic(ctx, 15, 2);
	if (nk_option_label(ctx, "Ignore Particles", lbm->outOfBoundsMode == LBM3D_1D_indices::CYCLE_ALL)) {
		lbm->outOfBoundsMode = LBM3D_1D_indices::IGNORE_PARTICLES;
	}
	if (nk_option_label(ctx, "Deactivate Particles", lbm->outOfBoundsMode == LBM3D_1D_indices::DEACTIVATE_PARTICLES)) {
		lbm->outOfBoundsMode = LBM3D_1D_indices::DEACTIVATE_PARTICLES;
	}
	if (nk_option_label(ctx, "Respawn Particles in Inlet", lbm->outOfBoundsMode == LBM3D_1D_indices::RESPAWN_PARTICLES_INLET)) {
		lbm->outOfBoundsMode = LBM3D_1D_indices::RESPAWN_PARTICLES_INLET;
	}
	*/

	nk_label_header(ctx, "Active Inlet Wall", false);
	nk_layout_row_dynamic(ctx, 15.0f, 2);
	nk_checkbox_label(ctx, "x left inlet", &lbm->xLeftInlet);
	nk_checkbox_label(ctx, "x right inlet", &lbm->xRightInlet);
	nk_checkbox_label(ctx, "y bottom inlet", &lbm->yBottomInlet);
	nk_checkbox_label(ctx, "y top inlet", &lbm->yTopInlet);
	nk_checkbox_label(ctx, "z left inlet", &lbm->zLeftInlet);
	nk_checkbox_label(ctx, "z right inlet", &lbm->zRightInlet);

	nk_label_header(ctx, "General");

	nk_checkbox_label(ctx, "Use Subgrid Model (Experimental)", &vars->useSubgridModel);

	nk_property_float(ctx, "Velocity Multiplier", 0.01f, &vars->lbmVelocityMultiplier, 10.0f, 0.01f, 0.01f);
	nk_checkbox_label(ctx, "Use Alternate Interpolation", &vars->lbmUseCorrectInterpolation);
	nk_checkbox_label(ctx, "Use Extended Collision Step (Unstable)", &vars->lbmUseExtendedCollisionStep);

	nk_label_header(ctx, "Editing");

	if (!lbm->isUnderEdit()) {
		if (nk_button_label(ctx, "Edit LBM Position")) {
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

		nk_property_float(ctx, "Scale", 1.0f, &lbm->scale, 1000.0f, 1.0f, 1.0f);

		if (nk_button_label(ctx, "Snap Corners to Ground")) {
			lbm->snapToGround();
		}

		if (nk_button_label(ctx, "Save Changes")) {

			//lbm->saveChanges(); // testing 
			lbm->stopEditing(true);
		}
		if (nk_button_label(ctx, "Cancel")) {
			lbm->stopEditing(false);
		}

	}

	nk_label_header(ctx, "Streamlines");
	//nk_layout_row_dynamic(ctx, 200, 1); // wrapping row

	//if (nk_group_begin(ctx, "Streamlines", NK_WINDOW_BORDER)) {

	if (!sps->initialized && !streamlineInitMode) {
		if (nk_button_label(ctx, "Use Streamlines")) {
			//sps->init();
			streamlineInitMode = true;
		}

	} else if (sps->initialized) {

		nk_checkbox_label(ctx, "Visible", &sps->visible);


		nk_layout_row(ctx, NK_STATIC, 15, 2, leftSidebarEditButtonRatio);

		if (nk_button_label(ctx, "Set Horizontal Line")) {
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

		if (nk_button_label(ctx, "Set Vertical Line")) {
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
			if (nk_button_label(ctx, "Deactivate Streamlines")) {
				sps->deactivate();
			}
		} else {
			if (nk_button_label(ctx, "Activate Streamlines")) {
				sps->activate();
			}
		}

		if (nk_button_label(ctx, "Reset")) {
			sps->reset();
		}


		nk_checkbox_label(ctx, "Live Cleanup (Recommended)", &sps->liveLineCleanup);

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

		nk_property_int(ctx, "Max Streamlines", 1, &sps->maxNumStreamlines, 10000, 1, 1);
		nk_property_int(ctx, "Max Streamline Length", 1, &sps->maxStreamlineLength, 1000, 1, 1);
		//nk_property_int(ctx, "streamline sampling", 1, &sps->sampling)

		if (nk_button_label(ctx, "Apply Settings")) {
			cout << "Initializing streamline data..." << endl;
			sps->init();
			streamlineInitMode = false;
		}
		if (nk_button_label(ctx, "Cancel")) {
			streamlineInitMode = false;
		}

	}
	//nk_group_end(ctx);

	//}













}

void UserInterface::constructLightingTab(int side) {


	nk_label_header(ctx, "Lighting", false);


	constructDirLightPositionPanel();

	nk_property_vec3(ctx, -1000000.0f, dirLight->focusPoint, 1000000.0f, 100.0f, 100.0f, "Sun Projection Focus Point");


	nk_label_header(ctx, "Projection Properties");

	nk_property_float(ctx, "Left:", -100000.0f, &dirLight->pLeft, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "Right:", -100000.0f, &dirLight->pRight, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "Bottom:", -100000.0f, &dirLight->pBottom, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "Top:", -100000.0f, &dirLight->pTop, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "Near:", 0.1f, &dirLight->pNear, 100000.0f, 10.0f, 10.0f);
	nk_property_float(ctx, "Far:", 1.0f, &dirLight->pFar, 1000000.0f, 1000.0f, 1000.0f);


	nk_label_header(ctx, "Exponential Variance Shadow Mapping");

	nk_checkbox_label(ctx, "Use Blur Pass", (int *)&evsm->useBlurPass);

	nk_property_float(ctx, "Shadow Bias", 0.0f, &evsm->shadowBias, 1.0f, 0.0001f, 0.0001f);
	nk_property_float(ctx, "Light Bleed Reduction", 0.0f, &evsm->lightBleedReduction, 1.0f, 0.01f, 0.01f);
	//nk_property_float(ctx, "variance min limit:", 0.0f, &evsm.varianceMinLimit, 1.0f, 0.0001f, 0.0001f);
	nk_property_float(ctx, "Exponent", 1.0f, &evsm->exponent, 42.0f, 0.1f, 0.1f);

	nk_checkbox_label(ctx, "Show Shadow Only", &evsm->shadowOnly);
	nk_property_float(ctx, "Shadow Intensity", 0.0f, &evsm->shadowIntensity, 1.0f, 0.01f, 0.01f);


	nk_label_header(ctx, "Fog");

	if (nk_combo_begin_label(ctx, VariableManager::getFogModeString(vars->fogMode).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
		nk_layout_row_dynamic(ctx, 15, 1);
		if (nk_combo_item_label(ctx, "Linear", NK_TEXT_LEFT)) {
			vars->fogMode = 0;
		}
		if (nk_combo_item_label(ctx, "Exponential", NK_TEXT_LEFT)) {
			vars->fogMode = 1;
		}
		nk_combo_end(ctx);

	}
	nk_property_float(ctx, "Fog Intensity", 0.0f, &vars->fogIntensity, 1.0f, 0.01f, 0.01f);


	if (vars->fogMode == eFogMode::LINEAR) {
		nk_property_float(ctx, "Fog Min Distance", 0.0f, &vars->fogMinDistance, 100000.0f, 1.0f, 10.0f);
		nk_property_float(ctx, "Fog Max Distance", 0.0f, &vars->fogMaxDistance, 100000.0f, 10.0f, 100.0f);
	} else {
		nk_property_float(ctx, "Fog Exp Falloff", 0.0f, &vars->fogExpFalloff, 1.0f, 0.01f, 0.01f);
	}
	nk_property_color_rgba(ctx, vars->fogColor);


	constructDirLightColorPanel();




}



#define USE_TERRAIN_MATERIAL_TOOLTIPS

void UserInterface::constructTerrainTab(int side) {

	nk_label_header(ctx, "Terrain");


	nk_checkbox_label(ctx, "Visible", &vars->heightMap->visible);
	nk_checkbox_label(ctx, "Visualize Normals", &vars->visualizeTerrainNormals);

	HeightMap *hm = vars->heightMap;

	if (nk_button_label(ctx, "Terrain Generator")) {
		openPopupWindow(terrainGeneratorWindowOpened);
	}

	nk_label(ctx, "Material Map", NK_TEXT_LEFT);
	if (nk_combo_begin_label(ctx, tryGetTextureFilename(hm->materialMap), nk_vec2(nk_widget_width(ctx), 200))) {
		nk_layout_row_dynamic(ctx, 15, 1);
		for (const auto& kv : *textures) {
			if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_LEFT)) {
				hm->materialMap = kv.second;
				nk_combo_close(ctx);
			}
		}
		nk_combo_end(ctx);
	}


	nk_label_header(ctx, "Global Normal Map");
	nk_property_float(ctx, "Mixing Ratio:", 0.0f, &hm->globalNormalMapMixingRatio, 1.0f, 0.01f, 0.01f);
	nk_property_float(ctx, "Tiling:", 1.0f, &hm->globalNormalMapTiling, 1000.0f, 0.1f, 0.1f);

	nk_label_header(ctx, "Grunge Map");
	nk_checkbox_label(ctx, "Use Grunge Map", &hm->useGrungeMap);
	nk_property_float(ctx, "Grunge Map min", 0.0f, &hm->grungeMapMin, 1.0f, 0.01f, 0.01f);
	nk_property_float(ctx, "Grunge Map Tiling", 1.0f, &hm->grungeMapTiling, 1000.0f, 0.1f, 0.1f);

	
	nk_label_header(ctx, "Texture Visualization");
	nk_checkbox_label(ctx, "Visualize Texture", &hm->visualizeTextureMode);

	if (hm->visualizeTextureMode) {
		if (nk_combo_begin_label(ctx, tryGetTextureFilename(hm->visTexture), nk_vec2(nk_widget_width(ctx), 200))) {
			nk_layout_row_dynamic(ctx, 15, 1);
			if (nk_combo_item_label(ctx, "NONE", NK_TEXT_LEFT)) {
				hm->visTexture = nullptr;
			}
			for (const auto& kv : *textures) {
				if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_LEFT)) {
					hm->visTexture = kv.second;
				}
			}
			nk_combo_end(ctx);
		}
	}

	nk_checkbox_label(ctx, "Normals Only", &vars->heightMap->showNormalsOnly);
	if (vars->heightMap->showNormalsOnly) {
		nk_property_int(ctx, "Normals Mode", 0, &vars->heightMap->normalsShaderMode, 2, 1, 1);
	}



	nk_label_header(ctx, "Materials", false);
	if (vars->terrainUsesPBR) {

		for (int i = 0; i < vars->heightMap->activeMaterialCount; i++) {

			if (nk_tree_push_id(ctx, NK_TREE_NODE, ("Material " + to_string(i)).c_str(), NK_MAXIMIZED, i)) {

				
#ifdef USE_TERRAIN_MATERIAL_TOOLTIPS

				struct nk_rect bounds;

				nk_layout_row_begin(ctx, NK_DYNAMIC, wh, 2);
				nk_layout_row_push(ctx, 0.1f);
				bounds = nk_widget_bounds(ctx);
				nk_label(ctx, "A", NK_TEXT_LEFT);
				nk_layout_row_push(ctx, 0.9f);
				constructTextureSelection(&vars->heightMap->pbrMaterials[i].albedo);
				nk_layout_row_end(ctx);

				if (nk_input_is_mouse_hovering_rect(ctx_in, bounds)) {
					nk_tooltip(ctx, "Albedo");
				}

				nk_layout_row_begin(ctx, NK_DYNAMIC, wh, 2);
				nk_layout_row_push(ctx, 0.1f);
				bounds = nk_widget_bounds(ctx);
				nk_label(ctx, "MR", NK_TEXT_LEFT);
				nk_layout_row_push(ctx, 0.9f);
				constructTextureSelection(&vars->heightMap->pbrMaterials[i].metallicRoughness);
				nk_layout_row_end(ctx);

				if (nk_input_is_mouse_hovering_rect(ctx_in, bounds)) {
					nk_tooltip(ctx, "Metallic Roughness");
				}

				nk_layout_row_begin(ctx, NK_DYNAMIC, wh, 2);
				nk_layout_row_push(ctx, 0.1f);
				bounds = nk_widget_bounds(ctx);
				nk_label(ctx, "NM", NK_TEXT_LEFT);
				nk_layout_row_push(ctx, 0.9f);
				constructTextureSelection(&vars->heightMap->pbrMaterials[i].normalMap);
				nk_layout_row_end(ctx);

				if (nk_input_is_mouse_hovering_rect(ctx_in, bounds)) {
					nk_tooltip(ctx, "Normal Map");
				}

				nk_layout_row_begin(ctx, NK_DYNAMIC, wh, 2);
				nk_layout_row_push(ctx, 0.1f);
				bounds = nk_widget_bounds(ctx);
				nk_label(ctx, "AO", NK_TEXT_LEFT);
				nk_layout_row_push(ctx, 0.9f);
				constructTextureSelection(&vars->heightMap->pbrMaterials[i].ao);
				nk_layout_row_end(ctx);

				if (nk_input_is_mouse_hovering_rect(ctx_in, bounds)) {
					nk_tooltip(ctx, "Ambient Occlusion");
				}
#else 
				nk_layout_row_dynamic(ctx, 15, 2);

				if (side == S_LEFT) {
					nk_label(ctx, "Albedo", NK_TEXT_LEFT);
					constructTextureSelection(&vars->heightMap->pbrMaterials[i].albedo);

					nk_label(ctx, "Metallic Roughness", NK_TEXT_LEFT);
					constructTextureSelection(&vars->heightMap->pbrMaterials[i].metallicRoughness);

					nk_label(ctx, "Normal Map", NK_TEXT_LEFT);
					constructTextureSelection(&vars->heightMap->pbrMaterials[i].normalMap);

					nk_label(ctx, "Ambient Occlusion", NK_TEXT_LEFT);
					constructTextureSelection(&vars->heightMap->pbrMaterials[i].ao);

				} else {
					constructTextureSelection(&vars->heightMap->pbrMaterials[i].albedo);
					nk_label(ctx, "Albedo", NK_TEXT_LEFT);

					constructTextureSelection(&vars->heightMap->pbrMaterials[i].metallicRoughness);
					nk_label(ctx, "Metallic Roughness", NK_TEXT_LEFT);

					constructTextureSelection(&vars->heightMap->pbrMaterials[i].normalMap);
					nk_label(ctx, "Normal Map", NK_TEXT_LEFT);

					constructTextureSelection(&vars->heightMap->pbrMaterials[i].ao);
					nk_label(ctx, "Ambient Occlusion", NK_TEXT_LEFT);

				}
#endif

				nk_layout_row_dynamic(ctx, wh, 1);
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
					if (nk_combo_item_label(ctx, "NONE", NK_TEXT_LEFT)) {
						vars->heightMap->materials[i].diffuseTexture = nullptr;
					}
					for (const auto& kv : *textures) {
						if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_LEFT)) {
							vars->heightMap->materials[i].diffuseTexture = kv.second;
						}
					}
					nk_combo_end(ctx);
				}

				nk_label(ctx, "Specular", NK_TEXT_LEFT);

				if (nk_combo_begin_label(ctx, vars->heightMap->materials[i].tryGetTextureFilename(Texture::eTextureMaterialType::SPECULAR).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
					nk_layout_row_dynamic(ctx, 15, 1);
					if (nk_combo_item_label(ctx, "NONE", NK_TEXT_LEFT)) {
						vars->heightMap->materials[i].specularMap = nullptr;
					}
					for (const auto& kv : *textures) {
						if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_LEFT)) {
							vars->heightMap->materials[i].specularMap = kv.second;
						}
					}
					nk_combo_end(ctx);
				}

				nk_label(ctx, "Normal Map", NK_TEXT_LEFT);

				if (nk_combo_begin_label(ctx, vars->heightMap->materials[i].tryGetTextureFilename(Texture::eTextureMaterialType::NORMAL_MAP).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
					nk_layout_row_dynamic(ctx, 15, 1);
					if (nk_combo_item_label(ctx, "NONE", NK_TEXT_LEFT)) {
						vars->heightMap->materials[i].normalMap = nullptr;
					}
					for (const auto& kv : *textures) {
						if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_LEFT)) {
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


	
	//if (nk_button_label(ctx, "Refresh LBM HEIGHTMAP")) {
	//	lbm->refreshHeightMap();
	//}
	//nk_property_int(ctx, "x offset", 0, &vars->terrainXOffset, 1000, 1, 1);
	//nk_property_int(ctx, "z offset", 0, &vars->terrainZOffset, 1000, 1, 1);


}

void UserInterface::constructTerrainGeneratorWindow() {
	if (terrainGeneratorWindowOpened) {
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
				terrainGeneratorWindowOpened = false;
			}
			if (nk_button_label(ctx, "Close")) {
				terrainGeneratorWindowOpened = false;
			}

			nk_end(ctx);
		} else {
			terrainGeneratorWindowOpened = false;
		}
	}
}








void UserInterface::constructSkyTab(int side) {


	nk_label_header(ctx, "Sky", false);


	constructDirLightPositionPanel();

	nk_layout_row_dynamic(ctx, 15, 1);


	nk_checkbox_label(ctx, "Skybox", &vars->drawSkybox);
	nk_checkbox_label(ctx, "Hosek", &vars->hosekSkybox);


	nk_property_double(ctx, "Turbidity", 1.0, &hosek->turbidity, 10.0, 0.1, 0.1f);
	nk_property_double(ctx, "Albedo", 0.0, &hosek->albedo, 1.0, 0.01, 0.01f);


	nk_property_double(ctx, "Horizon Offset", 0.001, &hosek->horizonOffset, 10.0, 0.001, 0.001f);
	nk_property_float(ctx, "Sun Intensity", 0.1f, &hosek->sunIntensity, 10.0f, 0.1f, 0.1f);
	nk_property_int(ctx, "Sun Exponent", 1, &hosek->sunExponent, 1024, 1, 1);


	nk_checkbox_label(ctx, "Recalculate Live", &hosek->liveRecalc);

	if (!hosek->liveRecalc) {
		if (nk_button_label(ctx, "Recalculate Model")) {
			hosek->update();
		}
	}


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







	nk_value_float(ctx, "Eta", (float)hosek->eta);
	nk_value_float(ctx, "Eta (degrees)", hosek->getElevationDegrees());

	nk_value_float(ctx, "Sun Theta", (float)hosek->sunTheta);
	nk_value_float(ctx, "Sun Theta (degrees)", glm::degrees((float)hosek->sunTheta));

	nk_checkbox_label(ctx, "Simulate sun", &vars->simulateSun);
	nk_checkbox_label(ctx, "Skip night", &dirLight->skipNightTime);
	nk_property_float(ctx, "Sun speed", 0.1f, &dirLight->circularMotionSpeed, 1000.0f, 0.1f, 0.1f);
	if (nk_option_label(ctx, "y axis", dirLight->rotationAxis == DirectionalLight::Y_AXIS)) {
		dirLight->rotationAxis = DirectionalLight::Y_AXIS;
	}
	if (nk_option_label(ctx, "z axis", dirLight->rotationAxis == DirectionalLight::Z_AXIS)) {
		dirLight->rotationAxis = DirectionalLight::Z_AXIS;
	}
	nk_property_float(ctx, "Rotation Radius:", 10000.0f, &dirLight->radius, 500000.0f, 100.0f, 100.0f);

	constructDirLightColorPanel();

}








void UserInterface::constructCloudVisualizationTab(int side) {


	nk_label_header(ctx, "Cloud Visualization", false);

	constructDirLightPositionPanel();



	nk_layout_row_dynamic(ctx, wh, 1);

	nk_checkbox_label(ctx, "Use Volumetric Rendering", &particleRenderer->useVolumetricRendering);

	if (particleRenderer->useVolumetricRendering) {

		nk_label_header(ctx, "Volumetric Rendering Settings");

		int prevNumSlices = particleRenderer->numSlices;
		nk_property_int(ctx, "Num Slices", 1, &particleRenderer->numSlices, particleRenderer->maxNumSlices, 1, 1);
		if (prevNumSlices != particleRenderer->numSlices) {
			particleRenderer->numDisplayedSlices = particleRenderer->numSlices;
		}

		nk_property_int(ctx, "Num Displayed Slices", 0, &particleRenderer->numDisplayedSlices, particleRenderer->numSlices, 1, 1);

		nk_value_int(ctx, "Batch Size", particleRenderer->batchSize);


		//nk_value_vec3(ctx, particleRenderer->lightVec, "Light Vector");
		//nk_value_vec3(ctx, particleRenderer->viewVec, "View Vector");
		//nk_value_float(ctx, "Dot product", glm::dot(particleRenderer->viewVec, particleRenderer->lightVec));
		nk_property_float(ctx, "Inversion Threshold", -1.0f, &particleRenderer->inversionThreshold, 1.0f, 0.01f, 0.01f);
		nk_value_bool(ctx, "Inverted Rendering", particleRenderer->invertedRendering);
		//nk_value_vec3(ctx, particleRenderer->halfVec, "Half Vector");


	} else {

		if (nk_button_label(ctx, "Sort Particles by Camera Distance")) {
			particleSystem->sortParticlesByDistance(camera->position, eSortPolicy::GREATER);
		}
	}

	constructFormBoxButtonPanel();

	nk_property_float(ctx, "Point Size", 0.1f, &particleSystem->pointSize, 100000.0f, 0.1f, 0.1f);
	nk_property_float(ctx, "Opacity Multiplier", 0.01f, &vars->opacityMultiplier, 10.0f, 0.01f, 0.01f);

	nk_checkbox_label(ctx, "Show Particles Below CCL", &particleRenderer->showParticlesBelowCCL);
	particleSystem->showHiddenParticles = particleRenderer->showParticlesBelowCCL;

	nk_property_color_rgb(ctx, vars->tintColor, "Tint Color:");


	if (particleRenderer->useVolumetricRendering) {

		nk_layout_row_dynamic(ctx, wh, 1);
		nk_property_float(ctx, "Shadow Alpha (100x)", 0.01f, &particleRenderer->shadowAlpha100x, 100.0f, 0.01f, 0.01f);


		nk_label_header(ctx, "Debugging/Testing Options");
		nk_property_int(ctx, "Shader set", 0, &particleRenderer->shaderSet, 2, 1, 1);
		particleRenderer->updateShaderSet();

		nk_property_int(ctx, "First Pass Shader Mode", 0, &particleRenderer->firstPassShaderMode, particleRenderer->numFirstPassShaderModes - 1, 1, 1);

		nk_property_int(ctx, "Second Pass Shader Mode", 0, &particleRenderer->secondPassShaderMode, particleRenderer->numSecondPassShaderModes - 1, 1, 1);


		nk_label(ctx, "Sprite Texture", NK_TEXT_LEFT);
		if (nk_combo_begin_label(ctx, getTextureName(particleRenderer->spriteTexture).c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
			nk_layout_row_dynamic(ctx, 15, 1);

			/*
			if (nk_combo_item_label(ctx, "NONE", NK_TEXT_CENTERED)) {
			particleRenderer->spriteTexture = nullptr; // not a very bright idea
			}
			*/
			for (int i = 0; i < particleRenderer->spriteTextures.size(); i++) {
				if (nk_combo_item_label(ctx, particleRenderer->spriteTextures[i]->filename.c_str(), NK_TEXT_LEFT)) {
					particleRenderer->spriteTexture = particleRenderer->spriteTextures[i];
				}
			}
			nk_combo_end(ctx);
		}


		nk_checkbox_label(ctx, "Cast Shadows", &vars->cloudsCastShadows);

		nk_property_float(ctx, "Cast Shadow Alpha Multiplier", 0.0f, &vars->cloudCastShadowAlphaMultiplier, 2.0f, 0.01f, 0.01f);

		nk_label_header(ctx, "Phase Function");

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
			nk_checkbox_label(ctx, "Multiply by Shadow Intensity", &particleRenderer->multiplyPhaseByShadow);
		}

		nk_label_header(ctx, "Atlas Texture");

		nk_checkbox_label(ctx, "Show Particle Texture Idx", &particleRenderer->showParticleTextureIdx);
		nk_checkbox_label(ctx, "Use Atlas Texture", &particleRenderer->useAtlasTexture);

		nk_label_header(ctx, "Light Texture Blurring");

		nk_checkbox_label(ctx, "Blur Light Texture", &particleRenderer->useBlurPass);
		nk_property_float(ctx, "Blur Amount", 0.0f, &particleRenderer->blurAmount, 10.0f, 0.01f, 0.01f);
	}


}






void UserInterface::constructDiagramControlsTab(int side) {


	nk_label_header(ctx, "STLP");


	if (nk_combo_begin_label(ctx, stlpDiagram->getTmpSoundingFilename().c_str(), nk_vec2(nk_widget_width(ctx), 200.0f))) {
		if (vars->soundingDataFilenames.empty()) {
			nk_label(ctx, "Empty...", NK_TEXT_LEFT);
		}

		for (int i = 0; i < vars->soundingDataFilenames.size(); i++) {
			nk_layout_row_dynamic(ctx, 15, 1);
			if (nk_combo_item_label(ctx, vars->soundingDataFilenames[i].c_str(), NK_TEXT_CENTERED)) {
				stlpDiagram->setTmpSoundingFilename(vars->soundingDataFilenames[i]);
			}
		}
		nk_combo_end(ctx);
	}
	
	nk_checkbox_label(ctx, "Use Orographic Parameters", &stlpDiagram->useOrographicParametersEdit);
	stlpDiagram->useOrographicParametersChanged = stlpDiagram->useOrographicParametersEdit != stlpDiagram->useOrographicParameters;

	nk_checkbox_label(ctx, "Sounding Curves Editing Enabled", &stlpDiagram->soundingCurveEditingEnabled);

	if (stlpDiagram->wasSoundingFilenameChanged()) {
		if (nk_button_label(ctx, "Load Sounding File")) {
			stlpDiagram->loadSoundingData();
			stlpDiagram->recalculateAll();
			stlpSimCUDA->uploadDataFromDiagramToGPU();
			particleSystem->clearVerticalVelocities();
		}
	}/* else if (stlpDiagram->wasDiagramChanged() || stlpDiagram->useOrographicParametersChanged) {*/
	if (nk_button_label(ctx, "Recalculate Parameters")) {
		stlpDiagram->recalculateParameters();
		stlpSimCUDA->uploadDataFromDiagramToGPU();
		particleSystem->clearVerticalVelocities();
	}
	//}



	if (nk_tree_push(ctx, NK_TREE_TAB, "Diagram Curves", NK_MINIMIZED)) {
		nk_layout_row_static(ctx, 15.0f, vars->rightSidebarWidth, 1);
		stlpDiagram->constructDiagramCurvesToolbar(ctx, this);
		nk_tree_pop(ctx);
	}

	nk_layout_row_static(ctx, 15.0f, vars->rightSidebarWidth, 1);


	if (nk_button_label(ctx, "Reset to Default")) {
		stlpDiagram->recalculateAll();
		stlpSimCUDA->uploadDataFromDiagramToGPU();
		particleSystem->clearVerticalVelocities();
	}

	if (nk_button_label(ctx, "Clear Vertical Velocities")) {
		particleSystem->clearVerticalVelocities();
	}

	nk_layout_row_dynamic(ctx, 15, 1);
	//nk_property_float(ctx, "Zoom", -1.0f, &vars->diagramProjectionOffset, 1.0f, 0.01f, 0.01f);

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

	nk_property_int(ctx, "Number of Profiles", 2, &stlpDiagram->numProfiles, 100, 1, 1.0f); // somewhere bug when only one profile -> FIX!

	nk_property_float(ctx, "Profile Range", -10.0f, &stlpDiagram->convectiveTempRange, 10.0f, 0.01f, 0.01f);

	nk_checkbox_label(ctx, "Show Particles in Diagram", &vars->drawOverlayDiagramParticles);
	if (vars->drawOverlayDiagramParticles) {

		nk_checkbox_label(ctx, "Synchronize with Active Particles", &particleSystem->synchronizeDiagramParticlesWithActiveParticles);
		
		if (!particleSystem->synchronizeDiagramParticlesWithActiveParticles) {
			if (nk_button_label(ctx, "Activate All")) {
				particleSystem->activateAllDiagramParticles();
			}
			if (nk_button_label(ctx, "Deactivate All")) {
				particleSystem->deactivateAllDiagramParticles();
			}
			nk_property_int(ctx, "Num Particles Drawn", 0, &particleSystem->numDiagramParticlesToDraw, particleSystem->numActiveParticles, 1, 1);
		}
		nk_property_color_rgb(ctx, particleSystem->diagramParticlesColor);
		nk_layout_row_dynamic(ctx, wh, 1);

	}

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

		float tmp = stlpDiagram->overlayDiagramResolution;
		float maxDiagramWidth = (float)((vars->screenWidth < vars->screenHeight) ? vars->screenWidth : vars->screenHeight);
		nk_slider_float(ctx, 10.0f, &stlpDiagram->overlayDiagramResolution, maxDiagramWidth, 1.0f);


		float prevX = stlpDiagram->overlayDiagramX;
		float prevY = stlpDiagram->overlayDiagramY;
		nk_property_float(ctx, "x:", 0.0f, &stlpDiagram->overlayDiagramX, vars->screenWidth - stlpDiagram->overlayDiagramResolution, 0.1f, 0.1f);
		nk_property_float(ctx, "y:", 0.0f, &stlpDiagram->overlayDiagramY, vars->screenHeight - stlpDiagram->overlayDiagramResolution, 0.1f, 0.1f);
		
		if (tmp != stlpDiagram->overlayDiagramResolution ||
			prevX != stlpDiagram->overlayDiagramX ||
			prevY != stlpDiagram->overlayDiagramY) {
			stlpDiagram->refreshOverlayDiagram((float)vars->screenWidth, (float)vars->screenHeight);
		}
	}

	//nk_checkbox_label(ctx, "Use CUDA", &vars->stlpUseCUDA);

	//nk_checkbox_label(ctx, "Apply LBM", &vars->applyLBM);
	//nk_property_int(ctx, "LBM step frame", 1, &vars->lbmStepFrame, 100, 1, 1);

	/*
	bounds = nk_widget_bounds(ctx);
	if (nk_input_is_mouse_hovering_rect(in, bounds)) {
	nk_tooltip(ctx, "This is a tooltip");
	}
	*/



	nk_checkbox_label(ctx, "Apply STLP", &vars->applySTLP);
	nk_property_int(ctx, "STLP step frame", 1, &vars->stlpStepFrame, 100, 1, 1);

	//nk_property_float(ctx, "Point size", 0.1f, &particleSystem->pointSize, 100.0f, 0.1f, 0.1f);
	//stlpSimCUDA->pointSize = stlpSim->pointSize;
	//nk_property_float(ctx, "Point size (CUDA)", 0.1f, &stlpSimCUDA->pointSize, 100.0f, 0.1f, 0.1f);
/*

	struct nk_colorf tintColor;
	tintColor.r = vars->tintColor.x;
	tintColor.g = vars->tintColor.y;
	tintColor.b = vars->tintColor.z;

	nk_property_color_rgb(ctx, vars->tintColor);*/
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
	nk_layout_row_static(ctx, 15.0f, vars->rightSidebarWidth, 1);
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



	//tintColor.r = vars->bgClearColor.x;
	//tintColor.g = vars->bgClearColor.y;
	//tintColor.b = vars->bgClearColor.z;

	//if (nk_combo_begin_color(ctx, nk_rgb_cf(tintColor), nk_vec2(nk_widget_width(ctx), 400))) {
	//	nk_layout_row_dynamic(ctx, 120, 1);
	//	tintColor = nk_color_picker(ctx, tintColor, NK_RGBA);
	//	nk_layout_row_dynamic(ctx, 10, 1);
	//	tintColor.r = nk_propertyf(ctx, "#R:", 0, tintColor.r, 1.0f, 0.01f, 0.005f);
	//	tintColor.g = nk_propertyf(ctx, "#G:", 0, tintColor.g, 1.0f, 0.01f, 0.005f);
	//	tintColor.b = nk_propertyf(ctx, "#B:", 0, tintColor.b, 1.0f, 0.01f, 0.005f);
	//	//tintColor.a = nk_propertyf(ctx, "#A:", 0, tintColor.a, 1.0f, 0.01f, 0.005f);
	//	vars->bgClearColor = glm::vec3(tintColor.r, tintColor.g, tintColor.b);
	//	nk_combo_end(ctx);
	//}



	nk_value_bool(ctx, "Tc Found", stlpDiagram->TcFound);
	nk_value_bool(ctx, "EL Found", stlpDiagram->ELFound);
	nk_value_bool(ctx, "CCL Found", stlpDiagram->CCLFound);
	nk_value_bool(ctx, "LCL Found", stlpDiagram->LCLFound);
	nk_value_bool(ctx, "LFC Found", stlpDiagram->LFCFound);
	nk_value_bool(ctx, "Orographic EL Found", stlpDiagram->orographicELFound);





}







void UserInterface::constructLBMDebugTab(int side) {


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

void UserInterface::constructSceneHierarchyTab(int side) {


	hierarchyIdCounter = 0;
	activeActors.clear();

	nk_label_header(ctx, "Scene Hierarchy", false);


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
		nk_layout_row_dynamic(ctx, wh, 1);
		/*
		struct nk_rect w;
		w = nk_layout_widget_bounds(ctx);

		//nk_layout_row_dynamic(ctx, 15, 1);
		nk_layout_row_begin(ctx, NK_STATIC, 15.0f, 2);
		nk_layout_row_push(ctx, w.w - 20.0f);
		*/
		nk_selectable_label(ctx, actor->name.c_str(), NK_TEXT_LEFT, &actor->selected);
		/*nk_layout_row_push(ctx, 15.0f);

		if (nk_button_symbol(ctx, actor->visible ? NK_SYMBOL_CIRCLE_SOLID : NK_SYMBOL_CIRCLE_OUTLINE)) {
			actor->visible = !actor->visible;
		}
		nk_layout_row_end(ctx);*/

	}
	if (actor->selected) {
		activeActors.push_back(actor);
	}
}

void UserInterface::constructParticleSystemTab(int side) {

	nk_label_header(ctx, "Particle System");


	if (nk_button_label(ctx, "Load Particles from File")) {
		openPopupWindow(loadParticlesWindowOpened);
		particleSystem->loadParticleSaveFiles();
	}
	if (nk_button_label(ctx, "Save Particles to File")) {
		openPopupWindow(saveParticlesWindowOpened);
	}

	nk_layout_row_dynamic(ctx, wh, 1);
	if (nk_button_label(ctx, "Activate All Particles")) {
		particleSystem->activateAllParticles();
	}
	if (nk_button_label(ctx, "Deactivate All Particles")) {
		particleSystem->deactivateAllParticles();
	}
	nk_property_int(ctx, "Active Particles", 0, &particleSystem->numActiveParticles, particleSystem->numParticles, 1000, 100);

	if (nk_button_label(ctx, "Reset on Terrain")) {
		particleSystem->refreshParticlesOnTerrain();
	}


}


void UserInterface::constructEmittersTab(int side) {

	nk_label_header(ctx, "Emitters");

	if (nk_button_label(ctx, "Add Emitter")) {
		openPopupWindow(emitterCreationWindowOpened);
	}

	if (ebm->isActive()) {

		if (nk_button_label(ctx, "Disable Brush Mode")) {
			ebm->setActive(false);
		}

		ebm->constructBrushSelectionUIPanel(ctx, this);


	} else {


		if (nk_button_label(ctx, "Enable Brush Mode")) {
			ebm->setActive(true);
		}


		vector<int> emitterIndicesToDelete;

		for (int i = 0; i < particleSystem->emitters.size(); i++) {
			Emitter *e = particleSystem->emitters[i];

			if (nk_tree_push_id(ctx, NK_TREE_NODE, Emitter::getEmitterName(e), NK_MINIMIZED, i)) {

				e->constructEmitterPropertiesTab(ctx, this);
				/*
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
				*/

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
	}
	nk_label_header(ctx, "Particle Settings");

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

void UserInterface::constructEmitterCreationWindow() {
	if (emitterCreationWindowOpened) {
		//cout << "Emitter creation window is open" << endl;
		float w = 500.0f;
		float h = 500.0f;
		if (nk_begin(ctx, "Emitter Creation", nk_rect((vars->screenWidth - w) / 2.0f, (vars->screenHeight - h) / 2.0f, w, h), NK_WINDOW_CLOSABLE | NK_WINDOW_BORDER | NK_WINDOW_DYNAMIC | NK_WINDOW_NO_SCROLLBAR)) {

			//cout << "Emitter Creation opened successfuly" << endl;

			nk_layout_row_dynamic(ctx, 15, 1);

			if (nk_combo_begin_label(ctx, Emitter::getEmitterTypeString(selectedEmitterType), nk_vec2(nk_widget_width(ctx), 400.0f))) {
				nk_layout_row_dynamic(ctx, 15, 1);

				for (int i = 0; i < Emitter::eEmitterType::_NUM_EMITTER_TYPES; i++) {
					if (nk_combo_item_label(ctx, Emitter::getEmitterTypeString(i), NK_TEXT_CENTERED)) {
						selectedEmitterType = i;
					}
				}
				nk_combo_end(ctx);
			}

			nk_layout_row_dynamic(ctx, 15, 1);

			bool closeWindowAfterwards = false;
			particleSystem->constructEmitterCreationWindow(ctx, this, selectedEmitterType, closeWindowAfterwards);

			if (closeWindowAfterwards) {
				emitterCreationWindowOpened = false;
			}


		} else {
			emitterCreationWindowOpened = false;
		}
		nk_end(ctx);

	}

}

void UserInterface::constructGeneralDebugTab(int side) {
	stringstream ss;

	nk_label_header(ctx, "Debug");

	/*
	ss.clear();
	ss << "Delta time: " << fixed << setprecision(2) << (deltaTime * 1000.0);

	//string fpsStr = "delta time: " + to_string(deltaTime * 1000.0);
	nk_label(ctx, ss.str().c_str(), NK_TEXT_CENTERED);
	*/
	stringstream().swap(ss);
	ss << "Delta Time: " << fixed << setprecision(4) << prevAvgDeltaTime << " [ms] (" << setprecision(0) << prevAvgFPS << " FPS)";
	nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);

	nk_label_header(ctx, "Basic STLP Parameters");

	// Quick info -> creation of the strings should be moved to the Diagram since it only changes when the diagram is changed
	stringstream().swap(ss);
	ss << "T_c: " << fixed << setprecision(0) << stlpDiagram->Tc.x << " [deg C] at " << stlpDiagram->Tc.y << " [hPa]";
	nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);

	stringstream().swap(ss);
	ss << "CCL: " << fixed << setprecision(0) << stlpDiagram->CCL.x << " [deg C] at " << stlpDiagram->CCL.y << " [hPa]";

	nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);

	stringstream().swap(ss);
	ss << "EL: " << fixed << setprecision(0) << stlpDiagram->EL.x << " [deg C] at " << stlpDiagram->EL.y << " [hPa]";
	nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);

	stringstream().swap(ss);
	ss << "Ground Pressure: " << stlpDiagram->P0 << " [hPa]";
	nk_label(ctx, ss.str().c_str(), NK_TEXT_LEFT);


	// OVERLAY TEXTURES

	nk_label_header(ctx, "Overlay Textures", false);
	vector<OverlayTexture *> *overlayTextures = TextureManager::getOverlayTexturesVectorPtr();

	for (int i = 0; i < overlayTextures->size(); i++) {

		if (nk_tree_push_id(ctx, NK_TREE_NODE, ("Overlay Texture " + to_string(i)).c_str(), NK_MAXIMIZED, i)) {
			nk_layout_row_dynamic(ctx, 15, 2);
			nk_checkbox_label(ctx, "Active", &(*overlayTextures)[i]->active);
			nk_checkbox_label(ctx, "Show Alpha", &(*overlayTextures)[i]->showAlphaChannel);
			nk_layout_row_dynamic(ctx, 15, 1);
			if (nk_combo_begin_label(ctx, (*overlayTextures)[i]->getBoundTextureName().c_str(), nk_vec2(nk_widget_width(ctx), 200))) {
				nk_layout_row_dynamic(ctx, 15, 1);
				if (nk_combo_item_label(ctx, "NONE", NK_TEXT_LEFT)) {
					(*overlayTextures)[i]->texture = nullptr;
				}
				for (const auto& kv : *textures) {
					if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_LEFT)) {
						(*overlayTextures)[i]->texture = kv.second;
					}
				}
				nk_combo_end(ctx);
			}
			nk_tree_pop(ctx);
		}
	}

	nk_label_header(ctx, "Camera Info", false);

	nk_layout_row_dynamic(ctx, 15, 3);

	nk_value_float(ctx, "x", camera->position.x);
	nk_value_float(ctx, "y", camera->position.y);
	nk_value_float(ctx, "z", camera->position.z);


	nk_label_header(ctx, "Lattice Dimensions", false);

	nk_layout_row_dynamic(ctx, 15, 3);

	nk_value_int(ctx, "w", vars->latticeWidth);
	nk_value_int(ctx, "h", vars->latticeHeight);
	nk_value_int(ctx, "d", vars->latticeDepth);

	nk_value_int(ctx, "w [m]", (int)(vars->latticeWidth * lbm->scale));
	nk_value_int(ctx, "h [m]", (int)(vars->latticeHeight * lbm->scale));
	nk_value_int(ctx, "d [m]", (int)(vars->latticeDepth * lbm->scale));


	nk_label_header(ctx, "Terrain Dimensions");

	nk_value_int(ctx, "Terrain Texture Width", vars->heightMap->width);
	nk_value_int(ctx, "Terrain Texture Height", vars->heightMap->height);

	nk_value_int(ctx, "Terrain World Width", (int)vars->heightMap->getWorldWidth());
	nk_value_int(ctx, "Terrain World Depth", (int)vars->heightMap->getWorldDepth());


	if (vars->viewportMode == eViewportMode::VIEWPORT_3D) {
		if (nk_checkbox_label(ctx, "VSync", &vars->vsync)) {
			glfwSwapInterval(vars->vsync);
		}
	}

	nk_property_color_rgb(ctx, vars->bgClearColor, "BG Color:");

	if (nk_button_label(ctx, "Recompile Shaders - EXPERIMENTAL")) {
		ShaderManager::loadShaders();
	}

}

void UserInterface::constructPropertiesTab(int side) {

	nk_label_header(ctx, "Properties");

	if (activeActors.empty()) {
		nk_layout_row_dynamic(ctx, 60, 1);
		nk_label_wrap(ctx, "Select objects in hierarchy to display their properties here...");
	}

	for (const auto &actor : activeActors) {

		nk_layout_row_dynamic(ctx, 15, 1);
		nk_label(ctx, actor->name.c_str(), NK_TEXT_LEFT);
		nk_checkbox_label(ctx, "Visible", &actor->visible);

		if (!actor->isChildOfRoot()) {
			if (nk_button_label(ctx, "Move up a Level (Unparent)")) {
				actor->unparent();
			}
		}
		if (nk_button_label(ctx, "Snap to Ground")) {
			actor->snapToGround(vars->heightMap);
		}


		nk_layout_row_dynamic(ctx, 250, 1);
		if (nk_group_begin_titled(ctx, to_string(hierarchyIdCounter).c_str(), "Transform", NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR)) {
			nk_layout_row_dynamic(ctx, 15, 1);
			nk_property_vec3(ctx, -1000000.0f, actor->transform.position, 1000000.0f, 0.01f, 0.1f, "Position");
			nk_property_vec3(ctx, 0.0f, actor->transform.rotation, 360.0f, 0.1f, 0.1f, "Rotation");
			nk_property_vec3(ctx, 0.0f, actor->transform.scale, 1000.0f, 0.1f, 0.1f, "Scale");

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

void UserInterface::constructViewTab(int side) {

	nk_label_header(ctx, "View Options");

	nk_layout_row_begin(ctx, NK_DYNAMIC, wh, 2);
	nk_layout_row_push(ctx, 0.5f);
	nk_value_float(ctx, "Camera Speed", camera->movementSpeed);

	nk_layout_row_push(ctx, 0.5f);
	nk_slider_float(ctx, 1.0f, &camera->movementSpeed, 10000.0f, 1.0f);
	nk_layout_row_end(ctx);


	nk_label_header(ctx, "Projection Settings");

	if (nk_combo_begin_label(ctx, vars->projectionMode == ORTHOGRAPHIC ? "Orthographic" : "Perspective", nk_vec2(nk_widget_width(ctx), 200.0f))) {
		nk_layout_row_dynamic(ctx, wh, 1);
		if (nk_combo_item_label(ctx,  "Orthographic", NK_TEXT_LEFT)) {
			vars->projectionMode = ORTHOGRAPHIC;
			if (vars->useFreeRoamCamera) {
				vars->prevUseFreeRoamCamera = vars->useFreeRoamCamera;
				vars->useFreeRoamCamera = vars->useFreeRoamCamera == 0;
			}
		}
		if (nk_combo_item_label(ctx, "Perspective", NK_TEXT_LEFT)) {
			vars->projectionMode = PERSPECTIVE;
			vars->useFreeRoamCamera = vars->prevUseFreeRoamCamera;
		}
		nk_combo_end(ctx);
	}

	/*
	nk_layout_row_dynamic(ctx, wh, 2);
	if (nk_option_label(ctx, "Orthographic", vars->projectionMode == ORTHOGRAPHIC)) {
		vars->projectionMode = ORTHOGRAPHIC;
		if (vars->useFreeRoamCamera) {
			vars->prevUseFreeRoamCamera = vars->useFreeRoamCamera;
			vars->useFreeRoamCamera = vars->useFreeRoamCamera == 0;
		}
	}
	if (nk_option_label(ctx, "Perspective", vars->projectionMode == PERSPECTIVE)) {
		vars->projectionMode = PERSPECTIVE;
		if (vars->useFreeRoamCamera != vars->prevUseFreeRoamCamera) {
			vars->useFreeRoamCamera = vars->prevUseFreeRoamCamera;
		}
	}
	*/

	if (vars->projectionMode == PERSPECTIVE) {
		nk_layout_row_begin(ctx, NK_DYNAMIC, wh, 2);
		nk_layout_row_push(ctx, 0.5f);
		nk_value_float(ctx, "FOV", vars->fov);

		nk_layout_row_push(ctx, 0.5f);
		nk_slider_float(ctx, 30.0f, &vars->fov, 120.0f, 1.0f);
		nk_layout_row_end(ctx);
	}

	if (vars->projectionMode == PERSPECTIVE) {
		nk_layout_row_dynamic(ctx, wh, 1);
		if (nk_checkbox_label(ctx, "Use Freeroam Camera", &vars->useFreeRoamCamera)) {
			vars->prevUseFreeRoamCamera = vars->useFreeRoamCamera;
		}
	}


	if (!vars->useFreeRoamCamera) {
		nk_label(ctx, "Camera Settings", NK_TEXT_CENTERED);
		if (nk_button_label(ctx, "Front View (I)")) {
			camera->setView(Camera::VIEW_FRONT);
		}
		if (nk_button_label(ctx, "Side View (O)")) {
			camera->setView(Camera::VIEW_SIDE);
		}
		if (nk_button_label(ctx, "Top View (P)")) {
			camera->setView(Camera::VIEW_TOP);
		}
	}

	constructWalkingPanel();


}

void UserInterface::constructSaveParticlesWindow() {
	if (saveParticlesWindowOpened) {
		float w = 500.0f;
		float h = 500.0f;
		if (nk_begin(ctx, "Save Particles to File", nk_rect((vars->screenWidth - w) / 2.0f, (vars->screenHeight - h) / 2.0f, w, h), NK_WINDOW_CLOSABLE | NK_WINDOW_BORDER | NK_WINDOW_DYNAMIC | NK_WINDOW_NO_SCROLLBAR)) {



			bool closeWindowAfterwards = false;
			particleSystem->constructSaveParticlesWindow(ctx, this, closeWindowAfterwards);

			if (closeWindowAfterwards) {
				saveParticlesWindowOpened = false;
			}


		} else {
			saveParticlesWindowOpened = false;
		}
		nk_end(ctx);

	}


}

void UserInterface::constructLoadParticlesWindow() {
	if (loadParticlesWindowOpened) {
		float w = 500.0f;
		float h = 500.0f;
		if (nk_begin(ctx, "Load Particles from File", nk_rect((vars->screenWidth - w) / 2.0f, (vars->screenHeight - h) / 2.0f, w, h), NK_WINDOW_CLOSABLE | NK_WINDOW_BORDER | NK_WINDOW_DYNAMIC | NK_WINDOW_NO_SCROLLBAR)) {

			bool closeWindowAfterwards = false;
			particleSystem->constructLoadParticlesWindow(ctx, this, closeWindowAfterwards);

			if (closeWindowAfterwards) {
				loadParticlesWindowOpened = false;
			}


		} else {
			loadParticlesWindowOpened = false;
		}
		nk_end(ctx);

	}

}


void UserInterface::constructDebugTab() {
}

void UserInterface::constructFavoritesMenu() {
	nk_layout_row_push(ctx, 120);
	if (nk_menu_begin_label(ctx, "Favorites", NK_TEXT_CENTERED, toolbarMenuSize)) {
		nk_layout_row_dynamic(ctx, wh, 1);

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
		nk_checkbox_label(ctx, "Cloud Cast Shadows", &vars->cloudsCastShadows);

		if (vars->fullscreen) {
			if (nk_button_label(ctx, "Windowed")) {
				setFullscreen(false);
			}
		} else {
			if (nk_button_label(ctx, "Fullscreen")) {
				setFullscreen(true);
			}
			
		}



		nk_menu_end(ctx);
	}
}

void UserInterface::constructDirLightPositionPanel() {
	nk_property_vec3(ctx, -1000000.0f, dirLight->position, 1000000.0f, 100.0f, 100.0f, "Sun Position");
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

void UserInterface::constructDirLightColorPanel() {

	nk_label_header(ctx, "Directional Light");

	nk_checkbox_label(ctx, "Use Sky Sun Color", &vars->useSkySunColor);
	if (vars->useSkySunColor) {
		nk_property_float(ctx, "Sky Sun Color Tint Intensity", 0.0f, &vars->skySunColorTintIntensity, 1.0f, 0.01f, 0.01f);
	} else {
		nk_property_color_rgb(ctx, dirLight->color);
	}
	nk_layout_row_dynamic(ctx, 15.0f, 1);
	nk_property_float(ctx, "Intensity (PBR Only)", 0.0f, &dirLight->intensity, 1000.0f, 0.01f, 0.01f);


}

void UserInterface::constructHUD() {
	nk_style_item wfb = ctx->style.window.fixed_background;
	ctx->style.window.fixed_background = nk_style_item_hide();


	if (nk_begin(ctx, "HUD", hudRect, NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_NOT_INTERACTIVE | NK_WINDOW_NO_INPUT)) {
		nk_layout_row_dynamic(ctx, 15.0f, 1);

		struct nk_color tmpColor = ctx->style.text.color;
		ctx->style.text.color = nk_rgb(190, 255, 160);


		stringstream ss;

		ss << fixed << setprecision(4) << prevAvgDeltaTime << " [ms]";
		nk_label(ctx, ss.str().c_str(), NK_TEXT_RIGHT);

		stringstream().swap(ss);
		ss << fixed << setprecision(0) << prevAvgFPS << " FPS";
		nk_label(ctx, ss.str().c_str(), NK_TEXT_RIGHT);

		ctx->style.text.color = tmpColor;

	}

	nk_end(ctx);



	ctx->style.window.fixed_background = wfb;

}

void UserInterface::constructTextureSelection(Texture **targetTexturePtr, string nullTextureNameOverride, bool useWidgetWidth) {
	if (nk_combo_begin_label(ctx, tryGetTextureFilename(*targetTexturePtr, nullTextureNameOverride), useWidgetWidth ? nk_vec2(nk_widget_width(ctx), standardTexSelectSize.y) : standardTexSelectSize)) {
		nk_layout_row_dynamic(ctx, 15.0f, 1);
		if (nk_combo_item_label(ctx, "NONE", NK_TEXT_LEFT)) {
			*targetTexturePtr = nullptr;
		}
		for (const auto& kv : *textures) {
			if (nk_combo_item_label(ctx, kv.second->filename.c_str(), NK_TEXT_LEFT)) {
				(*targetTexturePtr) = kv.second;
			}
		}
		nk_combo_end(ctx);
	}
}

void UserInterface::nk_property_string(nk_context * ctx, std::string & target, char *buffer, int bufferLength, int &length) {

	nk_layout_row_dynamic(ctx, 30.0f, 1);
	nk_flags event = nk_edit_string(ctx, NK_EDIT_SIMPLE, &buffer[0], &length, bufferLength, nk_filter_default);

	if (event & NK_EDIT_ACTIVATED) {
		vars->generalKeyboardInputEnabled = false;
	}
	if (event & NK_EDIT_DEACTIVATED) {
		vars->generalKeyboardInputEnabled = true;
	}
	buffer[length] = '\0';
	target = string(buffer);

}

void UserInterface::setButtonStyle(nk_context *ctx, bool active) {
	ctx->style.button = active ? activeButtonStyle : inactiveButtonStyle;
	
}

void UserInterface::refreshWidgets() {
	hudRect = nk_rect((float)(vars->screenWidth - vars->rightSidebarWidth - hudWidth), (float)vars->toolbarHeight, hudWidth, hudHeight);
}

void UserInterface::setFullscreen(bool useFullscreen) {
	vars->fullscreen = useFullscreen;
	if (vars->fullscreen) {
		GLFWmonitor *monitor = glfwGetPrimaryMonitor();
		const GLFWvidmode *mode = glfwGetVideoMode(monitor);
		glfwSetWindowMonitor(mainWindow, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
	} else {
		const GLFWvidmode *mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		glfwSetWindowMonitor(mainWindow, NULL, 10, 20, vars->windowWidth, vars->windowHeight, GLFW_DONT_CARE);
	}
}

void UserInterface::constructTauProperty() {
	nk_property_float(ctx, "Tau:", 0.5005f, &lbm->tau, 10.0f, 0.005f, 0.005f);
}

void UserInterface::constructWalkingPanel() {
	if (vars->useFreeRoamCamera) {
		FreeRoamCamera *fcam = (FreeRoamCamera*)camera;
		int wasWalking = fcam->walking;
		nk_checkbox_label(ctx, "Walking", &fcam->walking);
		if (!wasWalking && fcam->walking) {
			fcam->snapToGround();
			//fcam->movementSpeed = 1.4f;
		}

		nk_property_float(ctx, "Player Height", 0.0f, &fcam->playerHeight, 10.0f, 0.01f, 0.01f);
	}
}

const char *UserInterface::tryGetTextureFilename(Texture * tex, std::string nullTextureName) {
	if (tex == nullptr) {
		if (!nullTextureName.empty()) {
			return nullTextureName.c_str();
		} else {
			return "NONE";
		}
	} else {
		return tex->filename.c_str();
	}
}

void UserInterface::openPopupWindow(bool &target) {
	closeAllPopupWindows();
	target = true;
}

void UserInterface::closeAllPopupWindows() {
	aboutWindowOpened = false;
	terrainGeneratorWindowOpened = false;
	emitterCreationWindowOpened = false;
	saveParticlesWindowOpened = false;
	loadParticlesWindowOpened = false;
}

