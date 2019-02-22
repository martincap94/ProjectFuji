#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
out vec2 TexCoords;

uniform mat4 uProjection;
uniform mat4 uView;

void main()
{
    gl_Position = uProjection * uView * vec4(vertex.xy, 0.0, 1.0);
    //TexCoords = vertex.zw;
	TexCoords = vec2(vertex.z, 1.0 - vertex.w);
}  