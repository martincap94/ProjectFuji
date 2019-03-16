#pragma once


enum eProjectionMode {
	ORTHOGRAPHIC,
	PERSPECTIVE
};

/// Enum listing all possible LBM types. LBM2D_reindex and LBM3D_reindexed were deprecated, hence they are absent.
enum eLBMType {
	LBM2D,
	LBM3D
};

enum eSortPolicy {
	LESS,
	GREATER,
	LEQUAL,
	GEQUAL
};