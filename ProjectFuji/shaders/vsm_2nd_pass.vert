#version 330 core

in vec4 a_Vertex;
in vec3 a_Normal;

out vec3 v_Normal;
out vec4 v_Vertex;
out vec4 v_LightSpacePos;

uniform mat4  u_View;
uniform mat4  u_Projection;
uniform mat4  u_LightSpaceMatrix;

void main() {
    v_Vertex   = u_View * a_Vertex;
    v_Normal   = mat3(u_View) * a_Normal;


	v_LightSpacePos = u_LightSpaceMatrix * a_Vertex;

    gl_Position = u_Projection * v_Vertex;
}
