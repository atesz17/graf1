//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 130
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	in vec3 vertexColor;	    // variable input from Attrib Array selected by glBindAttribLocation
	out vec3 color;				// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 130
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
	vec4 operator+(const vec4& other) const
	{
		return vec4(v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2]);
	}
	vec4 operator-(const vec4& other) const
	{
		return vec4(v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2]);
	}
	vec4 operator*(const float num) const
	{
		return vec4(v[0] * num, v[1] * num, v[2] * num);
	}
	vec4 operator/(const float num) const
	{
		return vec4(v[0] / num, v[1] / num, v[2] / num);
	}
};

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
		wCx = 0;
		wCy = 0;
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class Triangle {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
	float phi;
public:
	Triangle() {
		Animate(0, 0, 0, 0);
	}

	void Create(float r, float g, float b) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { -0.65, 0, 0.65, 0, 0, 3 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords),  // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

		// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { r, g, b, r, g, b, r, g, b };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

		// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
		// Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t, float pPhi, float pWTX, float pWTY) {
		sx = 1 * pow(sinf(t), 2) + 0.5f; // *sinf(t); // pulzalas --> pow, hogy ne legyen negativ, +konstans, hogy ne tunjon el amikor 0
		sy = 1; // *cosf(t);
		wTx = pWTX; // 4 * cosf(t / 2);
		wTy = pWTY; // 4 * sinf(t / 2);
		phi = pPhi + t;
	}

	void Draw() {
		mat4 M(sx * cos(phi), -sx * sin(phi), 0, 0,
			-sy * -sin(phi), sy * cos(phi), 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = M * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};


const int MAX_CTRL_POINT_COUNT = 20; // 20 - 1 = 19, ami ugyanaz, mint az elso
const int RESOLUTION = 12; // gorbe felbontasa
const int FLOAT_IN_VBO = 5;
const float TENSION = -0.8f;

struct CMSpline
{
	GLuint vao, vbo;
	float vertexData[(MAX_CTRL_POINT_COUNT)*FLOAT_IN_VBO*RESOLUTION];
	int nVertices;
	vec4 ctrlPoints[MAX_CTRL_POINT_COUNT];
	int nCtrlPoints;
	float ts[MAX_CTRL_POINT_COUNT];
	CMSpline()
	{
		nVertices = 0;
		nCtrlPoints = 0;
	}
	void Create()
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}
	void Draw() {
		if (nCtrlPoints > 2) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_LOOP, 0, nVertices);
		}
	}
	void AddPoint(float cX, float cY) {
		if (nCtrlPoints >= MAX_CTRL_POINT_COUNT) return;

		// attranszformaljuk vilag koordinatarendszerbe (ezert kell az inverz)
		// Pinv --> Projekcios inv transzf, Vinv --> View
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();

		if (nCtrlPoints <= MAX_CTRL_POINT_COUNT)
		{
			AddCtrlPoint(wVertex.v[0], wVertex.v[1]);
			AddTrailingCtrlPoint();
		}
		else
		{
			return;
		}
		if (nCtrlPoints >= 4) 
		{
			nVertices = 0; // ujraszamoljuk
			for (int i = 0; i < nCtrlPoints - 1; i++)
			{
				for (int j = 0; j < RESOLUTION; j++)
				{
					float deltaTime = ts[i + 1] - ts[i];
					printf("%f\n", ts[i] + j * deltaTime / RESOLUTION);
					vec4 point = r(ts[i] + j * deltaTime / RESOLUTION);
					AddVertexPoint(point.v[0], point.v[1]);
				}
			}
		}
		
		// printInfo();

		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}
	void printInfo()
	{
		for (int i = 0; i < nCtrlPoints; i++)
		{
			printf("X: %f, Y: %f, t: %f\n", ctrlPoints[i].v[0], ctrlPoints[i].v[1], ts[i]);
			if (i != nCtrlPoints - 1)
			{
				for (int j = 0; j < RESOLUTION; j++)
				{
					printf("    X: %f, Y: %f\n",
						vertexData[i * RESOLUTION * FLOAT_IN_VBO + j*FLOAT_IN_VBO],
						vertexData[i * RESOLUTION * FLOAT_IN_VBO + j*FLOAT_IN_VBO + 1]);
				}
			}
		}
		printf("\nnCtrlPoints: %d\n", nCtrlPoints);
		printf("nVertices: %d\n\n", nVertices);
	}
	vec4 r(float t)
	{
		for (int i = 0; i < nCtrlPoints - 1; i++)
		{
			if (ts[i] <= t && t <= ts[i + 1])
			{
				if (i == 0)
				{
					return Hermite(
						ctrlPoints[0],
						vec4(),
						ts[0],
						ctrlPoints[1],
						Velocity(TENSION, ctrlPoints[0], ts[0], ctrlPoints[1], ts[1], ctrlPoints[2], ts[2]),
						ts[1],
						t);
				}
				else if (i + 2 == nCtrlPoints)
				{
					return Hermite(
						ctrlPoints[nCtrlPoints - 2],
						Velocity(TENSION, ctrlPoints[nCtrlPoints - 3], ts[nCtrlPoints - 3], ctrlPoints[nCtrlPoints - 2], ts[nCtrlPoints - 2], ctrlPoints[nCtrlPoints - 1], ts[nCtrlPoints - 1]),
						ts[nCtrlPoints - 2],
						ctrlPoints[nCtrlPoints - 1],
						vec4(),
						ts[nCtrlPoints - 1],
						t);
				}
				else
				{
					return Hermite(
						ctrlPoints[i],
						Velocity(TENSION, ctrlPoints[i - 1], ts[i - 1], ctrlPoints[i], ts[i], ctrlPoints[i + 1], ts[i + 1]),
						ts[i],
						ctrlPoints[i + 1],
						Velocity(TENSION, ctrlPoints[i], ts[i], ctrlPoints[i + 1], ts[i + 1], ctrlPoints[i + 2], ts[i + 2]),
						ts[i + 1],
						t);
				}
			}
		}
	}
	vec4 Hermite(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t)
	{
		vec4 a0 = p0;
		vec4 a1 = v0;
		vec4 a2 = (p1 - p0) * 3 / pow(t1 - t0, 2) - (v1 + v0 * 2) / (t1 - t0);
		vec4 a3 = (p0 - p1) * 2 / pow(t1 - t0, 3) + (v1 + v0) / pow(t1 - t0, 2);

		return a3 * pow(t - t0, 3) + a2 * pow(t - t0, 2) + a1 * (t - t0) + a0;
	}
	vec4 Velocity(float tension, vec4 r0, float t0, vec4 r1, float t1, vec4 r2, float t2)
	{
		vec4 velocity = ((r2 - r1) / (t2 - t1) + (r1 - r0) / (t1 - t0)) * ((1 - tension) / 2);
		return velocity;
	}
	void AddTrailingCtrlPoint()
	{
		if (nCtrlPoints == 1)
		{
			ctrlPoints[nCtrlPoints] = ctrlPoints[0];
			ts[nCtrlPoints] = ts[nCtrlPoints - 1] + 0.5f;
			nCtrlPoints++;
		}
		else
		{
			swapTrailingPoints();
		}
	}
	void swapTrailingPoints()
	{
		vec4 temp = ctrlPoints[nCtrlPoints - 2];
		ctrlPoints[nCtrlPoints - 2] = ctrlPoints[nCtrlPoints - 1];
		ctrlPoints[nCtrlPoints - 1] = temp;

		float time = ts[nCtrlPoints - 1];
		ts[nCtrlPoints - 2] = ts[nCtrlPoints - 1];
		ts[nCtrlPoints - 1] = time + 0.5f;
	}
	void AddCtrlPoint(float wX, float wY)
	{
		ctrlPoints[nCtrlPoints].v[0] = wX;
		ctrlPoints[nCtrlPoints].v[1] = wY;
		
		ts[nCtrlPoints] = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

		nCtrlPoints++;
	}
	void AddVertexPoint(float wX, float wY)
	{
		vertexData[5 * nVertices] = wX;
		vertexData[5 * nVertices + 1] = wY;
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 0; // blue
		nVertices++;
	}
};

const int STAR_VERTICES_COUNT = 7; // hany aga legyen a csillagnak
const float STAR_ROAD_TRIP_TIME = 3.0f; // mennyi ideig tart, mig megtesz egy teljes kort a csillag

struct Star
{
	Triangle parts[STAR_VERTICES_COUNT];
	CMSpline *spline;
	Star *mainStar;
	bool isActive;
	float mass;
	vec4 position; // cameranak kell majd

	Star(CMSpline *pSpline, Star* pMainStar, float pMass) // focsillagnak van spline-ja, nincs starja, mellekcsillagnak pont forditva
	{
		spline = pSpline;
		mainStar = pMainStar;
		mass = pMass;
		isActive = false;
	}
	void Create(float r, float g, float b)
	{
		for (int i = 0; i < STAR_VERTICES_COUNT; i++)
		{
			parts[i].Create(r, g, b);
		}
	}
	void Animate(float t)
	{
		if (spline)
		{
			if (spline->nCtrlPoints >= 4)
			{
				float deltaDegree = 360.0f / STAR_VERTICES_COUNT;
				for (int i = 0; i < STAR_VERTICES_COUNT; i++)
				{
					float phi = (i * deltaDegree) * M_PI / 180; // radian
					float time = getRelativeTime();
					position = spline->r(spline->ts[0] + time);
					parts[i].Animate(t, phi, position.v[0], position.v[1]);
				}
			}
		}
		else if (mainStar)
		{
			if (mainStar->isActive)
			{
				float deltaDegree = 360.0f / STAR_VERTICES_COUNT;
				for (int i = 0; i < STAR_VERTICES_COUNT; i++)
				{
					float phi = (i * deltaDegree) * M_PI / 180; // radian
					// itt jon a relativ tomegvonzas szamolasa
					parts[i].Animate(t, phi, 0, 0);
				}
			}
		}
	}
	void Draw()
	{
		if (spline)
		{
			if (spline->nCtrlPoints >= 4)
			{
				isActive = true; // signal a mellekcsillagoknak, hogy rajzolodjanak ki
				for (int i = 0; i < STAR_VERTICES_COUNT; i++)
				{
					parts[i].Draw();
				}
			}
		}
		else if (mainStar)
		{
			if (mainStar->isActive) // ha a main csillag ki van rajzolva
			{
				for (int i = 0; i < STAR_VERTICES_COUNT; i++)
				{
					parts[i].Draw();
				}
			}
		}
	}
	float getRelativeTime()
	{
		float segmentTotalTime = spline->ts[spline->nCtrlPoints - 1] - spline->ts[0];
		float relTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
		while (relTime >= segmentTotalTime)
		{
			relTime -= segmentTotalTime;
		}
		//relTime *= segmentTotalTime / STAR_ROAD_TRIP_TIME;
		return relTime;
	}
};

void updateCameraCoords(Camera *cam, Star *star)
{
	cam->wCx = star->position.v[0];
	cam->wCy = star->position.v[1];
}

// The virtual world: collection of two objects
//Triangle triangle;
CMSpline lineStrip;
Star star(&lineStrip, 0, 10);
Star littleOne(0, &star, 2);
bool isCameraFollowingStar = false;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	//triangle.Create();
	lineStrip.Create();
	star.Create(1, 1, 0); // 1 1 0 --> yellow
	littleOne.Create(0, 0, 1);

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect Attrib Arrays to input variables of the vertex shader
	glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
	glBindAttribLocation(shaderProgram, 1, "vertexColor");    // vertexColor gets values from Attrib Array 1

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	//triangle.Draw();
	lineStrip.Draw();
	star.Draw();
	littleOne.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	else if (key == 32) isCameraFollowingStar = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		lineStrip.AddPoint(cX, cY);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	if (isCameraFollowingStar)
	{
		updateCameraCoords(&camera, &star);
	}
	camera.Animate(sec);					// animate the camera
	//triangle.Animate(sec);					// animate the triangle object
	star.Animate(sec);
	littleOne.Animate(sec);
	glutPostRedisplay();					// redraw the scene
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

