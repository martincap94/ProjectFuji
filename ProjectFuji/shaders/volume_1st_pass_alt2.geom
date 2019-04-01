#version 400 core

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;

uniform mat4  u_View;
uniform mat4  u_Projection;

uniform vec3 u_CameraPos;
uniform float u_WorldPointSize;

const vec3 worldup = vec3(0.0, 1.0, 0.0);

out vec2 g_TexCoords;

void main() {
	vec3 pos = gl_in[0].gl_Position.xyz;
	mat4 VP = u_Projection * u_View;

	float tmpscale = u_WorldPointSize / 10.0;

	vec3 toCamera = normalize(u_CameraPos - pos);
	vec3 right = normalize(cross(toCamera, worldup)) * tmpscale;
	vec3 up = normalize(cross(right, toCamera)) * tmpscale;

	pos -= (right * 0.5);
	gl_Position = VP * vec4(pos, 1.0);
	g_TexCoords = vec2(0.0, 0.0);
	EmitVertex();

	pos += up;
	gl_Position = VP * vec4(pos, 1.0);
	g_TexCoords = vec2(0.0, 1.0);
	EmitVertex();

	pos -= up;
	pos += right;
	gl_Position = VP * vec4(pos, 1.0);
	g_TexCoords = vec2(1.0, 0.0);
	EmitVertex();

	pos += up;
	gl_Position = VP * vec4(pos, 1.0);
	g_TexCoords = vec2(1.0, 1.0);
	EmitVertex();

	EndPrimitive();

}


