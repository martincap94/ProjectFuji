#version 400 core

layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

in NormalData {
	vec3 normal;
} normalData[];

const float size = 2.0;

void generateLine(int index) {
	gl_Position = gl_in[index].gl_Position;
	EmitVertex();
	gl_Position = gl_in[index].gl_Position + vec4(normalData[index].normal, 0.0) * size;
	EmitVertex();
	EndPrimitive();
}

void main() {
	generateLine(0);
	generateLine(1);
	generateLine(2);
}
