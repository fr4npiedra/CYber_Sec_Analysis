<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateShootingsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('shootings', function (Blueprint $table) {
            $table->id();
            $table->string('name');
            $table->string('date');
            $table->string('manner_of_death');
            $table->string('armer');
            $table->integer('age');
            $table->string('gender');
            $table->string('race');
            $table->string('city');
            $table->string('state');
            $table->string('signs_of_mental_illness');
            $table->string('threat_level');
            $table->string('flee');
            $table->string('body_camera');
            $table->string('arms_category');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('shootings');
    }
}
